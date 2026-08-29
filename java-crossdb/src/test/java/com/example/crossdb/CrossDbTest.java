package com.example.crossdb;

import org.junit.jupiter.api.Test;

import javax.sql.DataSource;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

class CrossDbTest {

  private static final String JOIN_SQL =
      "SELECT u.name, COUNT(*) AS cnt, SUM(o.amount) AS total "
      + "FROM userdb.users u JOIN orderdb.orders o ON o.user_id = u.id "
      + "GROUP BY u.name ORDER BY u.name";

  private static Map<String, String> run(CrossDb db, String sql) throws SQLException {
    try (ResultSet rs = db.register("userdb", Fixtures.USERS)
        .register("orderdb", Fixtures.ORDERS).query(sql)) {
      Map<String, String> m = new LinkedHashMap<>();
      while (rs.next()) {
        m.put(rs.getString(1), rs.getInt(2) + "," + rs.getInt(3));
      }
      return m;
    }
  }

  @Test void crossDbJoinGroupBy() throws Exception {
    try (CrossDb db = new CrossDb()) {
      Map<String, String> m = run(db, JOIN_SQL);
      assertEquals("2,15", m.get("alice"));
      assertEquals("2,21", m.get("bob"));
      assertFalse(m.containsKey("carol"));
    }
  }

  @Test void bindJoinPushesInInsteadOfInnerFullScan() throws Exception {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    DataSource users = Recording.dataSource(Fixtures.USERS, sqls, new ArrayList<>());
    DataSource orders = Recording.dataSource(Fixtures.ORDERS, sqls, new ArrayList<>());
    Map<String, String> m;
    try (CrossDb db = new CrossDb(1000, 1_000_000L, 500, 2)) {
      try (ResultSet rs = db.register("userdb", users).register("orderdb", orders)
          .query(JOIN_SQL)) {
        m = new LinkedHashMap<>();
        while (rs.next()) {
          m.put(rs.getString(1), rs.getInt(2) + "," + rs.getInt(3));
        }
      }
    }
    assertEquals(Map.of("alice", "2,15", "bob", "2,21"), m);
    long inCount = sqls.stream().filter(s -> s.contains("IN (")).count();
    assertEquals(1, inCount, "应有且仅有 1 条 IN 下推: " + sqls);
    assertTrue(sqls.stream().anyMatch(s -> s.contains("IN (?,?)")),
        "两个 key 应合并进一个 IN 批次: " + sqls);
    assertEquals(2, sqls.size(), "内表 IN 查询 + 驱动侧全量拉取各一条: " + sqls);
  }

  @Test void compositeJoinConditionFallsBackToNative() throws Exception {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    DataSource users = Recording.dataSource(Fixtures.USERS, sqls, new ArrayList<>());
    DataSource orders = Recording.dataSource(Fixtures.ORDERS, sqls, new ArrayList<>());
    try (CrossDb db = new CrossDb()) {
      Map<String, String> m = new LinkedHashMap<>();
      try (ResultSet rs = db.register("userdb", users).register("orderdb", orders).query(
          "SELECT u.name, COUNT(*) AS cnt FROM userdb.users u JOIN orderdb.orders o "
          + "ON o.user_id = u.id AND o.amount > u.id GROUP BY u.name ORDER BY u.name")) {
        while (rs.next()) {
          m.put(rs.getString(1), rs.getString(2));
        }
      }
      assertEquals(Map.of("alice", "2", "bob", "1"), m);
      assertTrue(sqls.stream().noneMatch(s -> s.contains("IN (")),
          "复合条件不应触发 Bind Join: " + sqls);
    }
  }

  @Test void crossDbLeftJoinKeepsUnmatchedAndBindsInner() throws Exception {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    DataSource users = Recording.dataSource(Fixtures.USERS, sqls, new ArrayList<>());
    DataSource orders = Recording.dataSource(Fixtures.ORDERS, sqls, new ArrayList<>());
    try (CrossDb db = new CrossDb()) {
      Map<String, String> m = new LinkedHashMap<>();
      try (ResultSet rs = db.register("userdb", users)
          .register("orderdb", orders).query(
          "SELECT u.name, COUNT(o.id) AS cnt FROM userdb.users u "
          + "LEFT JOIN orderdb.orders o ON o.user_id = u.id GROUP BY u.name ORDER BY u.name")) {
        while (rs.next()) {
          m.put(rs.getString(1), rs.getString(2));
        }
      }
      assertEquals(Map.of("alice", "2", "bob", "2", "carol", "0"), m);
      long inCount = sqls.stream().filter(s -> s.contains(" IN (")).count();
      long fullCount = sqls.stream().filter(s -> !s.contains(" IN (")).count();
      assertEquals(1, inCount, "LEFT JOIN 内表应走 IN 下推: " + sqls);
      assertEquals(1, fullCount, "只有驱动侧一条拉取，内表不应全量拉取: " + sqls);
    }
  }

  @Test void whereClauseOnCrossDbJoinWorksAndPushesDown() throws Exception {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    DataSource users = Recording.dataSource(Fixtures.USERS, sqls, new ArrayList<>());
    DataSource orders = Recording.dataSource(Fixtures.ORDERS, sqls, new ArrayList<>());
    try (CrossDb db = new CrossDb()) {
      List<String> names = new ArrayList<>();
      try (ResultSet rs = db.register("userdb", users).register("orderdb", orders).query(
          "SELECT u.name FROM userdb.users u JOIN orderdb.orders o ON o.user_id = u.id "
          + "WHERE u.name = 'alice'")) {
        while (rs.next()) {
          names.add(rs.getString(1));
        }
      }
      assertEquals(List.of("alice", "alice"), names);
      assertTrue(sqls.stream().anyMatch(s -> s.contains("'alice'")),
          "过滤条件应下推到源库: " + sqls);
    }
  }

  @Test void transitivePredicatePushedToBothSides() throws Exception {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    DataSource users = Recording.dataSource(Fixtures.USERS, sqls, new ArrayList<>());
    DataSource orders = Recording.dataSource(Fixtures.ORDERS, sqls, new ArrayList<>());
    try (CrossDb db = new CrossDb()) {
      int n = 0;
      try (ResultSet rs = db.register("userdb", users).register("orderdb", orders).query(
          "SELECT u.name FROM userdb.users u JOIN orderdb.orders o ON o.user_id = u.id "
          + "WHERE u.id = 1")) {
        while (rs.next()) {
          n++;
        }
      }
      assertEquals(2, n);
      // WHERE u.id = 1 沿 join key 传递：orders 侧 SQL 应包含 user_id = 1
      assertTrue(sqls.stream().anyMatch(s -> s.contains("ORDERS") && s.contains("= 1")),
          "传递谓词应下推到 orders 侧: " + sqls);
    }
  }

  @Test void topNPushesOrderAndLimitIntoDriverSql() throws Exception {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    DataSource orders = Recording.dataSource(Fixtures.ORDERS, sqls, new ArrayList<>());
    try (CrossDb db = new CrossDb()) {
      List<Integer> ids = new ArrayList<>();
      try (ResultSet rs = db.register("userdb", Fixtures.USERS).register("orderdb", orders)
          .query("SELECT o.id FROM orderdb.orders o JOIN userdb.users u ON o.user_id = u.id "
              + "ORDER BY o.id DESC LIMIT 2")) {
        while (rs.next()) {
          ids.add(rs.getInt(1));
        }
      }
      assertEquals(List.of(103, 102), ids);
      String ordersSql = sqls.stream().filter(s -> s.contains("ORDERS")).findFirst().orElse("");
      assertTrue(ordersSql.contains("ORDER BY"), "驱动侧应带 ORDER BY: " + sqls);
      assertTrue(ordersSql.contains("LIMIT") || ordersSql.contains("FETCH"),
          "驱动侧应带 LIMIT/FETCH: " + sqls);
    }
  }

  @Test void compositeKeyBindJoinAcrossDbs() throws Exception {
    try (CrossDb db = new CrossDb()) {
      Map<String, String> m = new LinkedHashMap<>();
      try (ResultSet rs = db.register("credsdb", Fixtures.CREDS)
          .register("quotasdb", Fixtures.QUOTAS).query(
          "SELECT c.login, q.quota FROM credsdb.creds c JOIN quotasdb.quotas q "
          + "ON c.user_id = q.user_id AND c.tenant_id = q.tenant_id ORDER BY c.login")) {
        while (rs.next()) {
          m.put(rs.getString(1), rs.getString(2));
        }
      }
      assertEquals(Map.of("a1", "10", "a2", "5", "b1", "20"), m);
    }
  }

  @Test void safeModeRejectsBareScanButAllowsFiltered() throws Exception {
    try (CrossDb db = new CrossDb().safeMode()) {
      db.register("userdb", Fixtures.USERS).register("orderdb", Fixtures.ORDERS);
      assertThrows(CrossDbUnsafeQueryException.class,
          () -> db.query("SELECT id FROM userdb.users"));
      assertThrows(CrossDbUnsafeQueryException.class,
          () -> db.query("SELECT o.id FROM orderdb.orders o JOIN userdb.users u "
              + "ON o.user_id = u.id"));
      try (ResultSet rs = db.query("SELECT name FROM userdb.users WHERE id = 1")) {
        assertTrue(rs.next());
        assertEquals("alice", rs.getString(1));
      }
    }
  }

  @Test void queryTimeoutPropagatesToSources() throws Exception {
    List<String> props = Collections.synchronizedList(new ArrayList<>());
    DataSource users = Recording.dataSource(Fixtures.USERS, new ArrayList<>(), props);
    try (CrossDb db = new CrossDb(1000, 1_000_000L, 500, 2, 3)) {
      try (ResultSet rs = db.register("userdb", users).register("orderdb", Fixtures.ORDERS)
          .query("SELECT id FROM userdb.small")) {
        assertTrue(rs.next());
      }
    }
    assertTrue(props.contains("setQueryTimeout(3)"), "应设置 queryTimeout=3: " + props);
  }

  @Test void analyzeReportsSourceSqlAndNetworkRows() throws Exception {
    try (CrossDb db = new CrossDb()) {
      db.register("userdb", Fixtures.USERS).register("orderdb", Fixtures.ORDERS);
      String report = db.analyze(JOIN_SQL);
      assertTrue(report.contains("userdb") && report.contains("orderdb"), report);
      assertTrue(report.contains("networkRows"), report);
      assertTrue(report.contains("SELECT"), report);
      assertTrue(report.contains("bindJoin"), report);
    }
  }

  @Test void explainReturnsPlan() throws Exception {
    try (CrossDb db = new CrossDb()) {
      db.register("userdb", Fixtures.USERS).register("orderdb", Fixtures.ORDERS);
      assertTrue(db.explain(JOIN_SQL).contains("BindJoin"),
          "explain 应含 BindJoin 算子");
    }
  }

  @Test void sameDbJoinStillWorks() throws Exception {
    try (CrossDb db = new CrossDb()) {
      List<String> names = new ArrayList<>();
      try (ResultSet rs = db.register("userdb", Fixtures.USERS)
          .register("orderdb", Fixtures.ORDERS).query(
          "SELECT u.name FROM userdb.users u JOIN userdb.small s ON s.id = u.id "
          + "ORDER BY u.name")) {
        while (rs.next()) {
          names.add(rs.getString(1));
        }
      }
      assertEquals(List.of("alice", "bob"), names);
    }
  }

  @Test void rowLimitExactlyAtThresholdPasses() throws Exception {
    try (CrossDb db = new CrossDb(1000, 2, 1, 1)) {
      List<Integer> ids = new ArrayList<>();
      try (ResultSet rs = db.register("userdb", Fixtures.USERS)
          .register("orderdb", Fixtures.ORDERS).query("SELECT id FROM userdb.small")) {
        while (rs.next()) {
          ids.add(rs.getInt(1));
        }
      }
      assertEquals(List.of(1, 2), ids);
    }
  }

  @Test void rowLimitExceededRejected() throws Exception {
    try (CrossDb db = new CrossDb(1000, 2, 1, 1)) {
      db.register("userdb", Fixtures.USERS).register("orderdb", Fixtures.ORDERS);
      try {
        try (ResultSet rs = db.query("SELECT id FROM userdb.users")) {
          while (rs.next()) {
            // 只消费不读列；熔断应发生在第 3 次 next()
          }
        }
        fail("超过阈值必须拒绝执行");
      } catch (Throwable e) {
        Throwable root = e;
        while (root.getCause() != null) {
          root = root.getCause();
        }
        assertTrue(String.valueOf(root.getMessage()).contains("熔断"),
            "根因应是熔断信息: " + root);
      }
    }
  }

  @Test void fetchSizePropagatesToSources() throws Exception {
    List<String> props = Collections.synchronizedList(new ArrayList<>());
    DataSource users = Recording.dataSource(Fixtures.USERS, new ArrayList<>(), props);
    DataSource orders = Recording.dataSource(Fixtures.ORDERS, new ArrayList<>(), props);
    try (CrossDb db = new CrossDb(7, 1_000_000L, 500, 2)) {
      try (ResultSet rs = db.register("userdb", users).register("orderdb", orders)
          .query(JOIN_SQL)) {
        assertTrue(rs.next());
      }
    }
    assertTrue(props.contains("setFetchSize(7)"), "源库语句应设置 fetchSize=7: " + props);
  }

  @Test void invalidConfigRejected() throws Exception {
    assertThrows(IllegalArgumentException.class, () -> new CrossDb(0, 10, 1, 1));
    assertThrows(IllegalArgumentException.class, () -> new CrossDb(1, 0, 1, 1));
    assertThrows(IllegalArgumentException.class, () -> new CrossDb(1, 10, 0, 1));
    assertThrows(IllegalArgumentException.class, () -> new CrossDb(1, 10, 1, 0));
  }

  @Test void invalidSqlRejected() throws Exception {
    try (CrossDb db = new CrossDb()) {
      assertThrows(SQLException.class, () -> db.query("SELEC 1"));
      assertThrows(SQLException.class,
          () -> db.query("SELECT * FROM nosuchdb.nosuchtable"));
    }
  }
}
