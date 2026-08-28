package com.example.crossdb;

import org.apache.calcite.linq4j.Linq4j;
import org.apache.calcite.linq4j.Enumerable;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/** Bind Join 运行时：外表 key 分批 IN 下推到内表所在库，并发拉取后本地 hash 探测。
 *
 * <p>ponytail: 左表全量驻内存；左表百万行级时改落盘/外部排序。
 */
public final class BindJoinExec {
  private BindJoinExec() {}

  public static Enumerable<Object[]> join(Enumerable<Object[]> left, DataSource dataSource,
      String sqlPrefix, String keyRef, int leftKeyIdx, int rightKeyIdx,
      int rightFieldCount, int batchSize, int parallelism) {
    if (Boolean.getBoolean("crossdb.debug")) {
      System.err.println("BindJoinExec: join 开始 ds=" + dataSource.getClass().getName()
          + " sqlPrefix=" + sqlPrefix);
    }
    List<Object[]> leftRows = left.toList();
    Map<Object, List<Object[]>> right = new HashMap<>();
    List<List<Object>> batches = new ArrayList<>();
    Set<Object> distinct = new HashSet<>();
    for (Object[] l : leftRows) {
      Object key = l[leftKeyIdx];
      if (key == null || !distinct.add(key)) {
        continue;
      }
      if (batches.isEmpty() || batches.get(batches.size() - 1).size() == batchSize) {
        batches.add(new ArrayList<>(Math.min(batchSize, 64)));
      }
      batches.get(batches.size() - 1).add(key);
    }
    if (!batches.isEmpty()) {
      ExecutorService pool =
          Executors.newFixedThreadPool(Math.max(1, Math.min(parallelism, batches.size())));
      try {
        List<Future<Map<Object, List<Object[]>>>> futures = new ArrayList<>();
        for (List<Object> keys : batches) {
          futures.add(pool.submit(
              () -> fetchBatch(dataSource, sqlPrefix, keyRef, keys, rightKeyIdx, rightFieldCount)));
        }
        for (Future<Map<Object, List<Object[]>>> f : futures) {
          for (Map.Entry<Object, List<Object[]>> e : f.get().entrySet()) {
            right.computeIfAbsent(e.getKey(), k -> new ArrayList<>()).addAll(e.getValue());
          }
        }
      } catch (Exception e) {
        throw e instanceof RuntimeException re ? re : new RuntimeException(e);
      } finally {
        pool.shutdownNow();
      }
    }
    List<Object[]> out = new ArrayList<>();
    for (Object[] l : leftRows) {
      Object key = l[leftKeyIdx];
      List<Object[]> matches = key == null ? null : right.get(key);
      if (matches == null) {
        continue;
      }
      for (Object[] r : matches) {
        Object[] row = new Object[l.length + rightFieldCount];
        System.arraycopy(l, 0, row, 0, l.length);
        System.arraycopy(r, 0, row, l.length, rightFieldCount);
        out.add(row);
      }
    }
    return Linq4j.asEnumerable(out);
  }

  private static Map<Object, List<Object[]>> fetchBatch(DataSource dataSource, String sqlPrefix,
      String keyRef, List<Object> keys, int rightKeyIdx, int width) throws Exception {
    StringBuilder sql = new StringBuilder(sqlPrefix)
        .append(" WHERE ").append(keyRef).append(" IN (");
    for (int i = 0; i < keys.size(); i++) {
      sql.append(i == 0 ? "?" : ",?");
    }
    sql.append(')');
    Map<Object, List<Object[]>> result = new HashMap<>();
    try (Connection c = dataSource.getConnection();
        PreparedStatement st = c.prepareStatement(sql.toString())) {
      for (int i = 0; i < keys.size(); i++) {
        st.setObject(i + 1, keys.get(i));
      }
      try (ResultSet rs = st.executeQuery()) {
        while (rs.next()) {
          Object[] row = new Object[width];
          for (int i = 0; i < width; i++) {
            row[i] = rs.getObject(i + 1);
          }
          result.computeIfAbsent(rs.getObject(rightKeyIdx + 1), k -> new ArrayList<>()).add(row);
        }
      }
    }
    return result;
  }
}
