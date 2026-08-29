package com.example.crossdb;

import org.apache.calcite.linq4j.Linq4j;
import org.apache.calcite.linq4j.Enumerable;
import org.junit.jupiter.api.Test;

import javax.sql.DataSource;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class BindJoinExecTest {

  private static final String PREFIX =
      "SELECT * FROM (SELECT \"ID\", \"USER_ID\", \"AMOUNT\" FROM \"ORDERS\") AS \"T\"";
  private static final String[] KEY = {"\"T\".\"USER_ID\""};

  private static Enumerable<Object[]> left(Object[]... rows) {
    return Linq4j.asEnumerable(List.of(rows));
  }

  private static Enumerable<Object[]> singleKey(Enumerable<Object[]> left, DataSource ds,
      int batch, int par) {
    return BindJoinExec.join(left, ds, PREFIX, KEY, new int[]{0}, new int[]{1}, 3, batch, par,
        false, false);
  }

  private static DataSource recordingDs(List<String> sqls) {
    return Recording.dataSource(Fixtures.ORDERS, sqls, new ArrayList<>());
  }

  private static DataSource recordingQuotasDs(List<String> sqls) {
    return Recording.dataSource(Fixtures.QUOTAS, sqls, new ArrayList<>());
  }

  @Test void joinProducesHashJoinResult() {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    Enumerable<Object[]> out = singleKey(
        left(new Object[]{1, "alice"}, new Object[]{2, "bob"}, new Object[]{3, "carol"}),
        recordingDs(sqls), 500, 2);
    List<Object[]> rows = out.toList();
    Map<Integer, List<Integer>> byKey = new TreeMap<>();
    for (Object[] r : rows) {
      byKey.computeIfAbsent((Integer) r[0], k -> new ArrayList<>()).add((Integer) r[4]);
    }
    byKey.values().forEach(Collections::sort);
    assertEquals(Map.of(1, List.of(5, 10), 2, List.of(1, 20)), byKey);
    assertEquals(5, rows.get(0).length);
    assertEquals("alice", rows.get(0)[1]);
    assertEquals(1, sqls.size());
    assertTrue(sqls.get(0).contains("IN (?,?,?)"), sqls.get(0));
  }

  @Test void duplicateKeysDeduplicatedIntoOneBatch() {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    Enumerable<Object[]> out = singleKey(
        left(new Object[]{1, "a"}, new Object[]{1, "a2"}, new Object[]{2, "b"}),
        recordingDs(sqls), 500, 2);
    List<Object[]> rows = out.toList();
    assertEquals(1, sqls.size());
    assertTrue(sqls.get(0).contains("IN (?,?)"), sqls.get(0));
    // 左表 3 行（含重复 key），hash 探测后重复 key 行各拿全量匹配
    assertEquals(6, rows.size());
  }

  @Test void batchOneKeyPerQuery() {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    Enumerable<Object[]> out = singleKey(
        left(new Object[]{1, "a"}, new Object[]{2, "b"}, new Object[]{3, "c"}),
        recordingDs(sqls), 1, 2);
    assertEquals(4, out.toList().size());
    assertEquals(3, sqls.size());
    assertTrue(sqls.stream().allMatch(s -> s.contains("IN (?)")), String.valueOf(sqls));
  }

  @Test void nullLeftKeySkipped() {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    Enumerable<Object[]> out = singleKey(
        left(new Object[]{null, "x"}, new Object[]{1, "alice"}),
        recordingDs(sqls), 1, 1);
    assertEquals(2, out.toList().size());
    assertEquals(1, sqls.size());
  }

  @Test void emptyLeftIssuesNoQuery() {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    Enumerable<Object[]> out = singleKey(left(), recordingDs(sqls), 500, 2);
    assertTrue(out.toList().isEmpty());
    assertEquals(0, sqls.size());
  }

  @Test void batchQueryPropagatesSqlFailure() {
    DataSource broken = (DataSource) java.lang.reflect.Proxy.newProxyInstance(
        getClass().getClassLoader(), new Class<?>[]{DataSource.class},
        (proxy, method, args) -> {
          if (method.getName().equals("getConnection")) {
            throw new SQLException("boom");
          }
          return null;
        });
    assertThrows(RuntimeException.class, () -> singleKey(
        left(new Object[]{1, "a"}), broken, 500, 1).toList());
  }

  @Test void leftOuterPadsUnmatchedAndNullKey() {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    Enumerable<Object[]> out = BindJoinExec.join(
        left(new Object[]{1, "alice"}, new Object[]{3, "carol"}, new Object[]{null, "x"}),
        recordingDs(sqls), PREFIX, KEY, new int[]{0}, new int[]{1}, 3, 500, 1, true, false);
    List<Object[]> rows = out.toList();
    // alice 匹配 2 行；carol 与 null key 各补一行 NULL（宽 5，右侧 3 列全 null）
    assertEquals(4, rows.size());
    assertEquals("carol", rows.get(2)[1]);
    assertEquals(5, rows.get(2).length);
    assertEquals(null, rows.get(2)[2]);
    assertEquals(null, rows.get(2)[4]);
    assertEquals("x", rows.get(3)[1]);
    assertEquals(1, sqls.size());
  }

  @Test void multiColumnKeysOrFormEndToEnd() {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    String prefix =
        "SELECT * FROM (SELECT \"TENANT_ID\", \"USER_ID\", \"QUOTA\" FROM \"QUOTAS\") AS \"T\"";
    Enumerable<Object[]> out = BindJoinExec.join(
        left(new Object[]{100, 1, "a1"}, new Object[]{200, 1, "a2"}, new Object[]{100, 2, "b1"}),
        recordingQuotasDs(sqls), prefix,
        new String[]{"\"T\".\"TENANT_ID\"", "\"T\".\"USER_ID\""},
        new int[]{0, 1}, new int[]{0, 1}, 3, 500, 1, false, false);
    Map<String, Integer> quota = new TreeMap<>();
    for (Object[] r : out.toList()) {
      quota.put((String) r[2], (Integer) r[5]);
    }
    assertEquals(Map.of("a1", 10, "a2", 5, "b1", 20), quota);
    assertEquals(1, sqls.size());
    // OR 降级形态：(A = ? AND B = ?) OR (A = ? AND B = ?)
    assertTrue(sqls.get(0).contains("OR"), sqls.get(0));
    assertTrue(sqls.get(0).contains("AND"), sqls.get(0));
  }

  @Test void multiColumnKeysTupleInEndToEnd() {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    String prefix =
        "SELECT * FROM (SELECT \"TENANT_ID\", \"USER_ID\", \"QUOTA\" FROM \"QUOTAS\") AS \"T\"";
    Enumerable<Object[]> out = BindJoinExec.join(
        left(new Object[]{100, 1, "a1"}, new Object[]{200, 1, "a2"}),
        recordingQuotasDs(sqls), prefix,
        new String[]{"\"T\".\"TENANT_ID\"", "\"T\".\"USER_ID\""},
        new int[]{0, 1}, new int[]{0, 1}, 3, 500, 1, false, true);
    Map<String, Integer> quota = new TreeMap<>();
    for (Object[] r : out.toList()) {
      quota.put((String) r[2], (Integer) r[5]);
    }
    assertEquals(Map.of("a1", 10, "a2", 5), quota);
    assertTrue(sqls.get(0).contains(") IN (("), sqls.get(0));
  }

  @Test void buildWhereForms() {
    String[] one = {"\"T\".\"ID\""};
    assertEquals("\"T\".\"ID\" IN (?,?)", BindJoinExec.buildWhere(one, 2, true));
    String[] two = {"\"T\".\"A\"", "\"T\".\"B\""};
    assertEquals("(\"T\".\"A\", \"T\".\"B\") IN ((?,?),(?,?))",
        BindJoinExec.buildWhere(two, 2, true));
    assertEquals("(\"T\".\"A\" = ? AND \"T\".\"B\" = ?) OR (\"T\".\"A\" = ? AND \"T\".\"B\" = ?)",
        BindJoinExec.buildWhere(two, 2, false));
  }

  @Test void streamingFetchesBatchOnDemandOnly() {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    Enumerable<Object[]> out = singleKey(
        left(new Object[]{1, "a"}, new Object[]{2, "b"}, new Object[]{3, "c"}),
        recordingDs(sqls), 1, 1);
    // 只消费第一行：流式执行应只下发第 1 批，不把外表/输出全量物化
    org.apache.calcite.linq4j.Enumerator<Object[]> iter = out.enumerator();
    assertTrue(iter.moveNext());
    assertEquals("a", iter.current()[1]);
    assertEquals(1, sqls.size());
    iter.close();
  }
}
