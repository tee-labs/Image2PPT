package com.example.crossdb;

import org.h2.jdbcx.JdbcDataSource;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Main {
  public static void main(String[] args) throws Exception {
    JdbcDataSource users = ds("jdbc:h2:mem:users;DB_CLOSE_DELAY=-1");
    try (Connection c = users.getConnection(); Statement s = c.createStatement()) {
      s.execute("CREATE TABLE users(id INT PRIMARY KEY, name VARCHAR(50))");
      s.execute("INSERT INTO users VALUES (1,'alice'),(2,'bob'),(3,'carol')");
      s.execute("CREATE TABLE small(id INT PRIMARY KEY)");
      s.execute("INSERT INTO small VALUES (1),(2)");
    }
    JdbcDataSource orders = ds("jdbc:h2:mem:orders;DB_CLOSE_DELAY=-1");
    try (Connection c = orders.getConnection(); Statement s = c.createStatement()) {
      s.execute("CREATE TABLE orders(id INT PRIMARY KEY, user_id INT, amount INT)");
      s.execute("INSERT INTO orders VALUES (100,1,10),(101,2,20),(102,1,5),(103,2,1)");
    }
    JdbcDataSource creds = ds("jdbc:h2:mem:creds;DB_CLOSE_DELAY=-1");
    try (Connection c = creds.getConnection(); Statement s = c.createStatement()) {
      s.execute("CREATE TABLE creds(user_id INT, tenant_id INT, login VARCHAR(20))");
      s.execute("INSERT INTO creds VALUES (1,100,'a1'),(2,100,'b1'),(1,200,'a2'),(3,100,'c1')");
    }
    JdbcDataSource quotas = ds("jdbc:h2:mem:quotas;DB_CLOSE_DELAY=-1");
    try (Connection c = quotas.getConnection(); Statement s = c.createStatement()) {
      s.execute("CREATE TABLE quotas(tenant_id INT, user_id INT, quota INT)");
      s.execute("INSERT INTO quotas VALUES (100,1,10),(100,2,20),(200,1,5)");
    }
    String joinSql =
        "SELECT u.name, COUNT(*) AS cnt, SUM(o.amount) AS total " +
        "FROM userdb.users u JOIN orderdb.orders o ON o.user_id = u.id " +
        "GROUP BY u.name ORDER BY u.name";

    System.out.println("== A: 默认配置，跨库 JOIN + GROUP BY（下推校验）");
    Map<String, String> a = runJoin(new CrossDb(), users, orders, joinSql);
    check("2,15".equals(a.get("alice")), "alice cnt/total: " + a.get("alice"));
    check("2,21".equals(a.get("bob")), "bob cnt/total: " + a.get("bob"));
    check(!a.containsKey("carol"), "carol 不应出现");

    System.out.println("== B: Bind Join（batch=500, parallel=2），内表分批 IN 下推");
    List<String> usersSqls = new ArrayList<>();
    List<String> ordersSqls = new ArrayList<>();
    List<String> usersProps = new ArrayList<>();
    List<String> ordersProps = new ArrayList<>();
    Map<String, String> b = runJoin(new CrossDb(1000, 1_000_000L, 500, 2),
        recording(users, usersSqls, usersProps), recording(orders, ordersSqls, ordersProps),
        joinSql);
    check(b.equals(a), "Bind Join 结果应与原生计划一致: " + b);
    List<String> inSqls = new ArrayList<>();
    List<String> fullSqls = new ArrayList<>();
    for (String s : concat(usersSqls, ordersSqls)) {
      (s.contains(" IN (") ? inSqls : fullSqls).add(s);
    }
    check(inSqls.size() == 1 && fullSqls.size() == 1,
        "应有 1 条 IN 下推（内表批量查）+ 1 条驱动侧拉取，实际 in=" + inSqls + " full=" + fullSqls
            + " users=" + usersSqls + " orders=" + ordersSqls);
    check(inSqls.get(0).contains("IN (?,?)"), "两个 key 应合并进一个 IN 批次: " + inSqls.get(0));
    check(usersProps.stream().anyMatch(p -> p.equals("setFetchSize(1000)"))
        && ordersProps.stream().anyMatch(p -> p.equals("setFetchSize(1000)")),
        "两侧源库语句都应设置 fetchSize=1000，实际 " + usersProps + ordersProps);
    System.out.println("   内表收到: " + inSqls.get(0).replace('\n', ' '));
    System.out.println("   驱动侧: " + fullSqls.get(0).replace('\n', ' '));

    System.out.println("== B2: Bind Join 运行时（batch=1, parallel=2）多批次 + 并发拉取");
    List<String> directSqls = new ArrayList<>();
    List<String> directProps = new ArrayList<>();
    org.apache.calcite.linq4j.Enumerable<Object[]> joined = BindJoinExec.join(
        org.apache.calcite.linq4j.Linq4j.asEnumerable(List.of(
            (Object[]) new Object[]{1, "alice"}, new Object[]{2, "bob"}, new Object[]{3, "carol"})),
        Guarded.wrap(recording(orders, directSqls, directProps), 1000, 1_000_000L),
        "SELECT * FROM (SELECT \"ID\", \"USER_ID\", \"AMOUNT\" FROM \"ORDERS\") AS \"T\"",
        new String[]{"\"T\".\"USER_ID\""}, new int[]{0}, new int[]{1}, 3, 1, 2, false, true);
    int aliceRows = 0;
    int bobRows = 0;
    for (Object[] row : joined.toList()) {
      if ("alice".equals(row[1])) {
        aliceRows++;
      }
      if ("bob".equals(row[1])) {
        bobRows++;
      }
    }
    check(aliceRows == 2 && bobRows == 2, "应 alice 2 行 bob 2 行，实际 " + aliceRows + "/" + bobRows);
    check(directSqls.size() == 3 && directSqls.stream().allMatch(s -> s.contains("IN (?)")),
        "batch=1 应对 3 个 key 产生 3 条单 key IN 查询，实际: " + directSqls);
    check(directProps.stream().anyMatch(p -> p.equals("setMaxRows(1000001)")),
        "批量查询也应受熔断 maxRows 保护，实际: " + directProps);
    System.out.println("   内表收到 " + directSqls.size() + " 条单 key IN 查询（并发执行）");

    System.out.println("== C: 危险 SQL 熔断（rowLimit=2）");
    List<String> breakerProps = new ArrayList<>();
    List<String> breakerSqls = new ArrayList<>();
    try (CrossDb db = new CrossDb(1000, 2, 1, 1)) {
      db.register("userdb", recording(users, breakerSqls, breakerProps))
          .register("orderdb", orders);
      try (ResultSet rs = db.query("SELECT id FROM userdb.small")) {
        int n = 0;
        while (rs.next()) {
          n++;
        }
        check(n == 2, "恰好等于阈值的查询应完整返回，实际 " + n + " 行");
      }
      int n = 0;
      try {
        try (ResultSet rs = db.query("SELECT id FROM userdb.users")) {
          while (rs.next()) {
            n++;
          }
        }
        check(false, "超过阈值必须拒绝执行，却返回了 " + n + " 行");
      } catch (Throwable e) {
        Throwable root = e;
        while (root.getCause() != null) {
          root = root.getCause();
        }
        check(String.valueOf(root.getMessage()).contains("熔断"), "熔断信息: " + e.getMessage());
        System.out.println("   " + root.getMessage());
      }
    }
    check(breakerProps.contains("setMaxRows(3)"),
        "源库语句应设置 maxRows=阈值+1，实际: " + breakerProps);

    System.out.println("== D: fetchSize=1 流式拉取");
    List<String> fetchSqls = new ArrayList<>();
    List<String> fetchProps = new ArrayList<>();
    Map<String, String> d = runJoin(new CrossDb(1, 1_000_000L, 500, 4),
        recording(users, fetchSqls, fetchProps), orders, joinSql);
    check(d.equals(a), "fetchSize=1 结果应与默认配置一致: " + d);
    check(fetchProps.stream().anyMatch(p -> p.equals("setFetchSize(1)")),
        "源库语句应设置 fetchSize=1，实际: " + fetchProps);

    System.out.println("== E: WHERE 过滤跨库查询（执行器回归）");
    try (CrossDb db = new CrossDb()) {
      db.register("userdb", users).register("orderdb", orders);
      List<String> names = new ArrayList<>();
      try (ResultSet rs = db.query(
          "SELECT u.name FROM userdb.users u JOIN orderdb.orders o ON o.user_id = u.id "
          + "WHERE u.name = 'alice'")) {
        while (rs.next()) {
          names.add(rs.getString(1));
        }
      }
      check(names.size() == 2, "WHERE 过滤应返回 alice 两行: " + names);
    }

    System.out.println("== F: LEFT JOIN Bind Join（内表不再全量拉取，未匹配补 NULL）");
    List<String> leftUsersSqls = new ArrayList<>();
    List<String> leftOrdersSqls = new ArrayList<>();
    try (CrossDb db = new CrossDb()) {
      db.register("userdb", recording(users, leftUsersSqls, new ArrayList<>()))
          .register("orderdb", recording(orders, leftOrdersSqls, new ArrayList<>()));
      Map<String, Integer> counts = new LinkedHashMap<>();
      try (ResultSet rs = db.query(
          "SELECT u.name, COUNT(o.id) AS cnt FROM userdb.users u "
          + "LEFT JOIN orderdb.orders o ON o.user_id = u.id GROUP BY u.name ORDER BY u.name")) {
        while (rs.next()) {
          counts.put(rs.getString(1), rs.getInt(2));
        }
      }
      check(Integer.valueOf(2).equals(counts.get("alice"))
          && Integer.valueOf(2).equals(counts.get("bob"))
          && Integer.valueOf(0).equals(counts.get("carol")),
          "LEFT JOIN 结果应 alice=2 bob=2 carol=0: " + counts);
      check(leftOrdersSqls.stream().anyMatch(s -> s.contains(" IN ("))
              && leftOrdersSqls.stream().noneMatch(s -> !s.contains(" IN (")),
          "LEFT JOIN 内表应只收到 IN 查询、不再全量拉取: " + leftOrdersSqls);
      System.out.println("   内表收到: " + leftOrdersSqls.get(0).replace('\n', ' '));
    }

    System.out.println("== G: 复合键 Bind Join（tuple-IN）");
    try (CrossDb db = new CrossDb()) {
      db.register("credsdb", creds).register("quotasdb", quotas);
      Map<String, String> m = new LinkedHashMap<>();
      try (ResultSet rs = db.query(
          "SELECT c.login, q.quota FROM credsdb.creds c JOIN quotasdb.quotas q "
          + "ON c.user_id = q.user_id AND c.tenant_id = q.tenant_id ORDER BY c.login")) {
        while (rs.next()) {
          m.put(rs.getString(1), rs.getString(2));
        }
      }
      check("10".equals(m.get("a1")) && "20".equals(m.get("b1")) && "5".equals(m.get("a2"))
              && !m.containsKey("c1"),
          "复合键 join 应 a1=10 b1=20 a2=5 无 c1: " + m);
    }

    System.out.println("== H: Top-N 下推（ORDER BY + LIMIT 进驱动侧 SQL）");
    List<String> topNOrdersSqls = new ArrayList<>();
    try (CrossDb db = new CrossDb()) {
      db.register("userdb", users)
          .register("orderdb", recording(orders, topNOrdersSqls, new ArrayList<>()));
      List<Integer> ids = new ArrayList<>();
      try (ResultSet rs = db.query(
          "SELECT o.id FROM orderdb.orders o JOIN userdb.users u ON o.user_id = u.id "
          + "ORDER BY o.id DESC LIMIT 2")) {
        while (rs.next()) {
          ids.add(rs.getInt(1));
        }
      }
      check(ids.equals(List.of(103, 102)), "Top-N 应返回 [103,102]: " + ids);
      String ordersSql = String.join("\n", topNOrdersSqls);
      check(ordersSql.contains("ORDER BY") && (ordersSql.contains("LIMIT")
              || ordersSql.contains("FETCH")),
          "驱动侧 SQL 应带 ORDER BY + LIMIT/FETCH: " + topNOrdersSqls);
      System.out.println("   驱动侧: " + ordersSql.replace('\n', ' '));
    }

    System.out.println("== I: 传递谓词下推（WHERE 常量沿 join key 传到两侧）");
    List<String> transUsersSqls = new ArrayList<>();
    List<String> transOrdersSqls = new ArrayList<>();
    try (CrossDb db = new CrossDb()) {
      db.register("userdb", recording(users, transUsersSqls, new ArrayList<>()))
          .register("orderdb", recording(orders, transOrdersSqls, new ArrayList<>()));
      int n = 0;
      try (ResultSet rs = db.query(
          "SELECT u.name FROM userdb.users u JOIN orderdb.orders o ON o.user_id = u.id "
          + "WHERE u.id = 1")) {
        while (rs.next()) {
          n++;
        }
      }
      check(n == 2, "u.id=1 应有两行: " + n);
      String all = String.join(" | ", concat(transUsersSqls, transOrdersSqls));
      check(all.contains("= 1") || all.contains("=1"),
          "两侧源库 SQL 均应下推 u.id/user_id = 1: " + all);
      System.out.println("   users 侧: " + String.join(" ", transUsersSqls).replace('\n', ' '));
      System.out.println("   orders 侧: " + String.join(" ", transOrdersSqls).replace('\n', ' '));
    }

    System.out.println("== J: queryTimeout 传播");
    List<String> timeoutProps = new ArrayList<>();
    try (CrossDb db = new CrossDb(1000, 1_000_000L, 500, 2, 3)) {
      db.register("userdb", recording(users, new ArrayList<>(), timeoutProps))
          .register("orderdb", orders);
      try (ResultSet rs = db.query("SELECT id FROM userdb.small")) {
        check(rs.next(), "超时配置不影响正常查询");
      }
    }
    check(timeoutProps.stream().anyMatch(p -> p.equals("setQueryTimeout(3)")),
        "源库语句应设置 queryTimeout=3，实际: " + timeoutProps);

    System.out.println("== K: safeMode 全表拉取拦截");
    try (CrossDb db = new CrossDb().safeMode()) {
      db.register("userdb", users).register("orderdb", orders);
      try {
        db.query("SELECT id FROM userdb.users");
        check(false, "裸全表扫描必须被拦截");
      } catch (CrossDbUnsafeQueryException e) {
        System.out.println("   拦截: " + e.getMessage());
      }
      try (ResultSet rs = db.query("SELECT name FROM userdb.users WHERE id = 1")) {
        check(rs.next() && "alice".equals(rs.getString(1)), "带 WHERE 的查询应放行");
      }
    }

    System.out.println("== L: explain / analyze");
    try (CrossDb db = new CrossDb()) {
      db.register("userdb", users).register("orderdb", orders);
      check(db.explain(JOIN_SQL_FOR_REPORT).contains("BindJoin"),
          "explain 应含 BindJoin 算子");
      String report = db.analyze(JOIN_SQL_FOR_REPORT);
      check(report.contains("userdb") && report.contains("orderdb")
          && report.contains("networkRows") && report.contains("bindJoin"),
          "analyze 应含各库 SQL/网络行数/bindJoin 统计:\n" + report);
      System.out.println(report);
    }

    System.out.println("SELF-CHECK OK: WHERE/LIMIT 执行器、Bind Join（LEFT/复合键/流式）、"
        + "Top-N 与传递谓词下推、熔断、超时传播、safeMode、explain/analyze 全部通过");
  }

  private static final String JOIN_SQL_FOR_REPORT =
      "SELECT u.name, SUM(o.amount) FROM userdb.users u "
      + "JOIN orderdb.orders o ON o.user_id = u.id GROUP BY u.name";

  private static List<String> concat(List<String> a, List<String> b) {
    List<String> all = new ArrayList<>(a);
    all.addAll(b);
    return all;
  }

  private static Map<String, String> runJoin(CrossDb db, javax.sql.DataSource users,
      javax.sql.DataSource orders, String sql) throws Exception {
    try (db; ResultSet rs = db.register("userdb", users).register("orderdb", orders).query(sql)) {
      Map<String, String> result = new LinkedHashMap<>();
      while (rs.next()) {
        String row = rs.getInt(2) + "," + rs.getInt(3);
        result.put(rs.getString(1), row);
        System.out.println("   " + rs.getString(1) + "  " + row);
      }
      return result;
    }
  }

  private static JdbcDataSource ds(String url) {
    JdbcDataSource d = new JdbcDataSource();
    d.setUrl(url);
    return d;
  }

  private static javax.sql.DataSource recording(javax.sql.DataSource delegate,
      List<String> sqls, List<String> props) {
    return Recording.dataSource(delegate, sqls, props);
  }

  private static void check(boolean ok, String message) {
    if (!ok) {
      throw new AssertionError("SELF-CHECK FAILED: " + message);
    }
  }
}
