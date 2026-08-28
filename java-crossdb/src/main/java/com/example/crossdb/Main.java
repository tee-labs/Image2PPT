package com.example.crossdb;

import org.h2.jdbcx.JdbcDataSource;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
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
      (s.contains("IN (") ? inSqls : fullSqls).add(s);
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
        "\"T\".\"USER_ID\"", 0, 1, 3, 1, 2);
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

    System.out.println("SELF-CHECK OK: 跨库 JOIN/GROUP BY、Bind Join 分批 IN 下推、行数熔断、流式拉取全部通过");
  }

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
    return (javax.sql.DataSource) Proxy.newProxyInstance(Main.class.getClassLoader(),
        new Class<?>[]{javax.sql.DataSource.class},
        (proxy, method, args) -> {
          if (method.getName().equals("getConnection")) {
            Connection c = (Connection) invoke(method, delegate, args);
            return (Connection) Proxy.newProxyInstance(Main.class.getClassLoader(),
                new Class<?>[]{Connection.class},
                (p2, m2, a2) -> {
                  Object r = invoke(m2, c, a2);
                  if (r instanceof Statement
                      && (m2.getName().startsWith("prepare") || m2.getName().equals("createStatement"))) {
                    if (m2.getName().startsWith("prepare") && a2 != null && a2.length > 0) {
                      sqls.add(String.valueOf(a2[0]));
                    }
                    Class<?> iface =
                        m2.getName().equals("createStatement") ? Statement.class : java.sql.PreparedStatement.class;
                    return statementRecording((Statement) r, iface, sqls, props);
                  }
                  return r;
                });
          }
          return invoke(method, delegate, args);
        });
  }

  private static Statement statementRecording(Statement target, Class<?> iface,
      List<String> sqls, List<String> props) {
    return (Statement) Proxy.newProxyInstance(Main.class.getClassLoader(),
        new Class<?>[]{iface},
        (proxy, method, args) -> {
          Object r = invoke(method, target, args);
          String n = method.getName();
          if ((n.equals("executeQuery") || n.equals("execute")) && args != null && args.length > 0) {
            sqls.add(String.valueOf(args[0]));
          }
          if ((n.equals("setFetchSize") || n.equals("setMaxRows")) && args != null && args.length > 0) {
            props.add(n + "(" + args[0] + ")");
          }
          return r;
        });
  }

  private static Object invoke(Method method, Object target, Object[] args) throws Exception {
    try {
      return method.invoke(target, args);
    } catch (InvocationTargetException e) {
      Throwable cause = e.getCause();
      if (cause instanceof Exception) {
        throw (Exception) cause;
      }
      throw e;
    }
  }

  private static void check(boolean ok, String message) {
    if (!ok) {
      throw new AssertionError("SELF-CHECK FAILED: " + message);
    }
  }
}
