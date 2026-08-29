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
  private static final String KEY = "\"T\".\"USER_ID\"";

  private static Enumerable<Object[]> left(Object[]... rows) {
    return Linq4j.asEnumerable(List.of(rows));
  }

  private static DataSource recordingDs(List<String> sqls) {
    return Recording.dataSource(Fixtures.ORDERS, sqls, new ArrayList<>());
  }

  /** 按左表 key 归组 amount，顺序无关断言。 */
  private static Map<Integer, List<Integer>> amountsByKey(Enumerable<Object[]> out) {
    Map<Integer, List<Integer>> byKey = new TreeMap<>();
    for (Object[] r : out.toList()) {
      byKey.computeIfAbsent((Integer) r[0], k -> new ArrayList<>()).add((Integer) r[4]);
    }
    byKey.values().forEach(Collections::sort);
    return byKey;
  }

  @Test void joinProducesHashJoinResult() {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    Enumerable<Object[]> out = BindJoinExec.join(
        left(new Object[]{1, "alice"}, new Object[]{2, "bob"}, new Object[]{3, "carol"}),
        recordingDs(sqls), PREFIX, KEY, 0, 1, 3, 500, 2);
    Map<Integer, List<Integer>> byKey = amountsByKey(out);
    assertEquals(Map.of(1, List.of(5, 10), 2, List.of(1, 20)), byKey);
    List<Object[]> rows = out.toList();
    assertEquals(5, rows.get(0).length);
    assertEquals("alice", rows.get(0)[1]);
    assertEquals(1, sqls.size());
    assertTrue(sqls.get(0).contains("IN (?,?,?)"), sqls.get(0));
  }

  @Test void duplicateKeysDeduplicatedIntoOneBatch() {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    Enumerable<Object[]> out = BindJoinExec.join(
        left(new Object[]{1, "a"}, new Object[]{1, "a2"}, new Object[]{2, "b"}),
        recordingDs(sqls), PREFIX, KEY, 0, 1, 3, 500, 2);
    assertEquals(1, sqls.size());
    assertTrue(sqls.get(0).contains("IN (?,?)"), sqls.get(0));
    // 左表 3 行（含重复 key），hash 探测后重复 key 行各拿全量匹配
    assertEquals(6, out.toList().size());
  }

  @Test void batchOneKeyPerQuery() {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    Enumerable<Object[]> out = BindJoinExec.join(
        left(new Object[]{1, "a"}, new Object[]{2, "b"}, new Object[]{3, "c"}),
        recordingDs(sqls), PREFIX, KEY, 0, 1, 3, 1, 2);
    assertEquals(3, sqls.size());
    assertTrue(sqls.stream().allMatch(s -> s.contains("IN (?)")), String.valueOf(sqls));
    assertEquals(4, out.toList().size());
  }

  @Test void nullLeftKeySkipped() {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    Enumerable<Object[]> out = BindJoinExec.join(
        left(new Object[]{null, "x"}, new Object[]{1, "alice"}),
        recordingDs(sqls), PREFIX, KEY, 0, 1, 3, 1, 1);
    assertEquals(1, sqls.size());
    assertEquals(2, out.toList().size());
  }

  @Test void emptyLeftIssuesNoQuery() {
    List<String> sqls = Collections.synchronizedList(new ArrayList<>());
    Enumerable<Object[]> out = BindJoinExec.join(left(), recordingDs(sqls),
        PREFIX, KEY, 0, 1, 3, 500, 2);
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
    assertThrows(RuntimeException.class, () -> BindJoinExec.join(
        left(new Object[]{1, "a"}), broken, PREFIX, KEY, 0, 1, 3, 500, 1));
  }
}
