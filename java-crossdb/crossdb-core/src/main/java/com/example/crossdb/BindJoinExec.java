package com.example.crossdb;

import org.apache.calcite.linq4j.AbstractEnumerable;
import org.apache.calcite.linq4j.Enumerable;
import org.apache.calcite.linq4j.Enumerator;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/** Bind Join 运行时（流水线式流式）：外表游标按窗口流式读取，每满一批 distinct key
 * 即异步下推 {@code WHERE (k1,k2) IN ((?,?),(?,?))...}（或 OR 降级）到内表库并发拉取，
 * 驱动侧内存占用 O(batchSize × parallelism) + 内表哈希表，输出按外表顺序流式 yield。
 *
 * <p>join 类型：INNER / LEFT（outer=true，未匹配外表行右侧补 NULL）/ RIGHT（innerFirst，
 * 外表与内表角色互换，行序仍保持 [原左 ++ 原右]）/ FULL（outer=true + 外表耗尽后对内表
 * 发起一次 NOT IN 反连接，补出未匹配的内表行）。
 *
 * <p>sortedByKey=true 时外表按 join key 有序，每窗口合并前淘汰「key 小于本窗口最小 key」
 * 的哈希旧条目，哈希表内存从 O(distinct keys) 降到 O(窗口 keys)。
 *
 * <p>ponytail: FULL 的反连接把全部外表 distinct key 攒在内存并一次性下发（分块 NOT IN），
 * 未匹配内表行一次性物化（受 rowLimit 熔断封顶）；如需流式反连接，改半连接两阶段执行。
 */
public final class BindJoinExec {
  private BindJoinExec() {}

  /** 兼容旧签名：无淘汰、无 RIGHT/FULL 形态。 */
  public static Enumerable<Object[]> join(Enumerable<Object[]> left, DataSource dataSource,
      String sqlPrefix, String[] keyCols, int[] leftKeys, int[] rightKeys, int width,
      int batchSize, int parallelism, boolean outer, boolean tupleIn) {
    return join(left, dataSource, sqlPrefix, keyCols, leftKeys, rightKeys, width,
        batchSize, parallelism, outer, tupleIn, false, false, false, 0);
  }

  /** 跨库 join 执行入口（由生成代码与单测直接调用）。
   *
   * @param left      外表（驱动侧）Enumerable
   * @param dataSource 内表所在库
   * @param sqlPrefix  内表可下推子树的 SELECT ... FROM (...) AS "T" 前缀
   * @param keyCols    内表 join key 的带引号列引用（如 "T"."USER_ID"），长度 = key 数
   * @param leftKeys   外表行的 key 列下标
   * @param rightKeys  内表行（SELECT * 结果）的 key 列下标
   * @param width      内表行宽度
   * @param batchSize  每批 distinct key 数上限（同时是窗口行数上限）
   * @param parallelism 并发批次数
   * @param outer      true=LEFT/RIGHT/FULL（未匹配的外表行补 NULL）
   * @param tupleIn    true=多列用 (k1,k2) IN ((?,?),...)；false=降级为 OR 组
   * @param sortedByKey 外表按 join key 有序（启用每窗口哈希淘汰）
   * @param innerFirst true=输出行为 [内表列 ++ 外表列]（RIGHT 交换后恢复原始列序）
   * @param full       true=外表耗尽后补内表未匹配行（FULL JOIN）
   * @param leftWidth  外表行宽度（FULL 补 NULL 左侧用）
   */
  public static Enumerable<Object[]> join(Enumerable<Object[]> left, DataSource dataSource,
      String sqlPrefix, String[] keyCols, int[] leftKeys, int[] rightKeys, int width,
      int batchSize, int parallelism, boolean outer, boolean tupleIn, boolean sortedByKey,
      boolean innerFirst, boolean full, int leftWidth) {
    Stats stats = Stats.ACTIVE;
    return new AbstractEnumerable<>() {
      @Override public Enumerator<Object[]> enumerator() {
        return new Enumerator<Object[]>() {
          final Iterator<Object[]> src = left.iterator();
          final Map<List<Object>, List<Object[]>> hash = new HashMap<>();
          final Deque<Pending> inflight = new ArrayDeque<>();
          final Set<List<Object>> queued = new HashSet<>();
          ExecutorService pool;
          Pending current;
          Iterator<Object[]> currentRows;
          List<Object[]> pendingMatches;
          Iterator<Object[]> remainder;
          boolean remainderDone;
          boolean evict = sortedByKey;
          Object[] currentLeft;
          Object[] ready;
          Object[] currentVal;

          class Pending {
            final List<Object[]> rows;
            final List<List<Object>> keys;
            final Future<Map<List<Object>, List<Object[]>>> future;

            Pending(List<Object[]> rows, List<List<Object>> keys,
                Future<Map<List<Object>, List<Object[]>>> future) {
              this.rows = rows;
              this.keys = keys;
              this.future = future;
            }
          }

          @Override public boolean moveNext() {
            if (ready == null) {
              try {
                advance();
              } catch (RuntimeException e) {
                shutdown();
                throw e;
              }
            }
            if (ready == null) {
              return false;
            }
            currentVal = ready;
            ready = null;
            return true;
          }

          @Override public Object[] current() {
            return currentVal;
          }

          @Override public void reset() {
            throw new UnsupportedOperationException();
          }

          @Override public void close() {
            shutdown();
          }

          private boolean advance() {
            while (true) {
              if (ready != null) {
                return true;
              }
              if (remainder != null) {
                if (!remainder.hasNext()) {
                  return false;
                }
                ready = padInner(remainder.next());
                return true;
              }
              if (pendingMatches != null) {
                if (pendingMatches.isEmpty()) {
                  pendingMatches = null;
                  continue;
                }
                ready = combine(currentLeft,
                    pendingMatches.remove(pendingMatches.size() - 1));
                return true;
              }
              if (current != null && currentRows.hasNext()) {
                probe();
                continue;
              }
              if (current != null) {
                current = null;
                currentRows = null;
                continue;
              }
              if (!nextBatch()) {
                if (full && !remainderDone) {
                  remainderDone = true;
                  remainder = fetchRemainder();
                  continue;
                }
                return false;
              }
            }
          }

          private void probe() {
            Object[] l = currentRows.next();
            List<Object> key = keyOf(l);
            List<Object[]> m = key == null ? null : hash.get(key);
            if (outer && (m == null || m.isEmpty())) {
              ready = pad(l);
              return;
            }
            if (m != null && !m.isEmpty()) {
              currentLeft = l;
              pendingMatches = new ArrayList<>(m);
            }
          }

          /** 拉起下一批：先用在途批次补满并行度（拉取与迭代重叠），再取队首合并进哈希表。 */
          private boolean nextBatch() {
            pump();
            if (inflight.isEmpty()) {
              return false;
            }
            Pending p = inflight.poll();
            if (p.future != null) {
              Map<List<Object>, List<Object[]>> fetched;
              try {
                fetched = p.future.get();
              } catch (Exception e) {
                shutdown();
                throw e instanceof RuntimeException re ? re : new RuntimeException(e);
              }
              evictBefore(p);
              long n = 0;
              for (Map.Entry<List<Object>, List<Object[]>> e : fetched.entrySet()) {
                hash.computeIfAbsent(e.getKey(), k -> new ArrayList<>()).addAll(e.getValue());
                n += e.getValue().size();
              }
              if (stats != null) {
                stats.batch(p.keys.size(), n);
              }
            }
            current = p;
            currentRows = p.rows.iterator();
            return true;
          }

          /** 外表流式读取：每凑满 batchSize 个 distinct key 或行数即打包异步下发。 */
          private void pump() {
            while (src.hasNext() && inflight.size() < parallelism) {
              List<Object[]> window = new ArrayList<>();
              Set<List<Object>> windowKeys = new LinkedHashSet<>();
              while (src.hasNext()) {
                Object[] l = src.next();
                window.add(l);
                List<Object> k = keyOf(l);
                if (k != null && queued.add(k)) {
                  windowKeys.add(k);
                }
                if (window.size() >= batchSize || windowKeys.size() >= batchSize) {
                  break;
                }
              }
              if (window.isEmpty()) {
                continue;
              }
              Future<Map<List<Object>, List<Object[]>>> f = null;
              if (!windowKeys.isEmpty()) {
                if (pool == null) {
                  pool = Executors.newFixedThreadPool(
                      Math.max(1, Math.min(parallelism, 8)), r -> {
                        Thread t = new Thread(r, "crossdb-bind");
                        t.setDaemon(true);
                        return t;
                      });
                }
                f = pool.submit(() -> fetchBatch(dataSource, sqlPrefix, keyCols, tupleIn,
                    new ArrayList<>(windowKeys), rightKeys, width));
              }
              inflight.add(new Pending(window, new ArrayList<>(windowKeys), f));
            }
          }

          private void shutdown() {
            if (pool != null) {
              pool.shutdownNow();
              pool = null;
            }
          }

          /** 外表按 key 有序时，合并前淘汰「key 小于本批最小 key」的哈希旧条目：
           * 有序流不会再回到更小的 key，这些条目不可能再被探测。 */
          private void evictBefore(Pending p) {
            if (!evict || p.keys.isEmpty()) {
              return;
            }
            try {
              List<Object> minKey = p.keys.get(0);
              hash.keySet().removeIf(k -> compareKeys(k, minKey) < 0);
            } catch (ClassCastException e) {
              evict = false; // key 不可比较：保守关闭淘汰，宁多占内存不错杀匹配
            }
          }

          /** FULL JOIN 收尾：内表中 key 不属于任何外表 key 的行（NOT IN 分块反连接）。 */
          private Iterator<Object[]> fetchRemainder() {
            try {
              return fetchRemainderRows(dataSource, sqlPrefix, keyCols, tupleIn,
                  new ArrayList<>(queued), width, batchSize).iterator();
            } catch (RuntimeException e) {
              shutdown();
              throw e;
            } catch (Exception e) {
              shutdown();
              throw new RuntimeException(e);
            }
          }

          List<Object> keyOf(Object[] row) {
            Object[] vals = new Object[leftKeys.length];
            for (int i = 0; i < leftKeys.length; i++) {
              vals[i] = row[leftKeys[i]];
            }
            List<Object> key = Arrays.asList(vals);
            return key.contains(null) ? null : key;
          }

          Object[] combine(Object[] l, Object[] r) {
            Object[] row = new Object[l.length + width];
            if (innerFirst) {
              System.arraycopy(r, 0, row, 0, width);
              System.arraycopy(l, 0, row, width, l.length);
            } else {
              System.arraycopy(l, 0, row, 0, l.length);
              System.arraycopy(r, 0, row, l.length, width);
            }
            return row;
          }

          Object[] pad(Object[] l) {
            Object[] row = new Object[l.length + width];
            if (innerFirst) {
              System.arraycopy(l, 0, row, width, l.length);
            } else {
              System.arraycopy(l, 0, row, 0, l.length);
            }
            return row;
          }

          Object[] padInner(Object[] r) {
            Object[] row = new Object[leftWidth + width];
            System.arraycopy(r, 0, row, leftWidth, width);
            return row;
          }
        };
      }
    };
  }

  private static Map<List<Object>, List<Object[]>> fetchBatch(DataSource dataSource,
      String sqlPrefix, String[] keyCols, boolean tupleIn, List<List<Object>> keys,
      int[] rightKeys, int width) throws Exception {
    StringBuilder sql = new StringBuilder(sqlPrefix).append(" WHERE ")
        .append(buildWhere(keyCols, keys.size(), tupleIn));
    Map<List<Object>, List<Object[]>> result = new HashMap<>();
    try (Connection c = dataSource.getConnection();
        PreparedStatement st = c.prepareStatement(sql.toString())) {
      int p = 0;
      for (List<Object> key : keys) {
        for (Object v : key) {
          st.setObject(++p, v);
        }
      }
      try (ResultSet rs = st.executeQuery()) {
        while (rs.next()) {
          Object[] row = new Object[width];
          for (int i = 0; i < width; i++) {
            row[i] = rs.getObject(i + 1);
          }
          Object[] kvals = new Object[rightKeys.length];
          for (int i = 0; i < rightKeys.length; i++) {
            kvals[i] = rs.getObject(rightKeys[i] + 1);
          }
          result.computeIfAbsent(Arrays.asList(kvals), k -> new ArrayList<>()).add(row);
        }
      }
    }
    return result;
  }

  /** 按字典序比较 key（Comparable 元素；不可比较时抛 CCE，由调用方降级）。 */
  @SuppressWarnings("unchecked")
  static int compareKeys(List<Object> a, List<Object> b) {
    for (int i = 0; i < a.size(); i++) {
      int c = ((Comparable<Object>) a.get(i)).compareTo(b.get(i));
      if (c != 0) {
        return c;
      }
    }
    return 0;
  }

  /** FULL JOIN 反连接：{@code WHERE NOT (key IN 分块1) AND NOT (key IN 分块2)...}；
   * 外表无非空 key 时内表全量即为未匹配。行数受 rowLimit 熔断封顶（语句级 maxRows）。 */
  private static List<Object[]> fetchRemainderRows(DataSource dataSource, String sqlPrefix,
      String[] keyCols, boolean tupleIn, List<List<Object>> keys, int width, int chunk)
      throws Exception {
    StringBuilder sql = new StringBuilder(sqlPrefix);
    if (!keys.isEmpty()) {
      List<String> nots = new ArrayList<>();
      for (int i = 0; i < keys.size(); i += Math.max(1, chunk)) {
        List<List<Object>> part = keys.subList(i, Math.min(keys.size(), i + Math.max(1, chunk)));
        nots.add("NOT (" + buildWhere(keyCols, part.size(), tupleIn) + ")");
      }
      sql.append(" WHERE ").append(String.join(" AND ", nots));
    }
    List<Object[]> rows = new ArrayList<>();
    try (Connection c = dataSource.getConnection();
        PreparedStatement st = c.prepareStatement(sql.toString())) {
      int p = 0;
      for (List<Object> key : keys) {
        for (Object v : key) {
          st.setObject(++p, v);
        }
      }
      try (ResultSet rs = st.executeQuery()) {
        while (rs.next()) {
          Object[] row = new Object[width];
          for (int i = 0; i < width; i++) {
            row[i] = rs.getObject(i + 1);
          }
          rows.add(row);
        }
      }
    }
    return rows;
  }

  /** 生成 IN 下推 WHERE 子句：单列 {@code c IN (?,?)}；多列 tuple
   * {@code (c1,c2) IN ((?,?),(?,?))} 或 OR 降级 {@code (c1=? AND c2=?) OR (...)}。 */
  static String buildWhere(String[] cols, int nValues, boolean tupleIn) {
    if (cols.length == 1) {
      return cols[0] + " IN (" + placeholders(nValues) + ")";
    }
    if (tupleIn) {
      String group = "(" + placeholders(cols.length) + ")";
      return "(" + String.join(", ", cols) + ") IN ("
          + String.join(",", java.util.Collections.nCopies(nValues, group)) + ")";
    }
    String group = "(" + String.join(" AND ",
        Arrays.stream(cols).map(c -> c + " = ?").toArray(String[]::new)) + ")";
    return String.join(" OR ", java.util.Collections.nCopies(nValues, group));
  }

  private static String placeholders(int n) {
    return String.join(",", java.util.Collections.nCopies(n, "?"));
  }
}
