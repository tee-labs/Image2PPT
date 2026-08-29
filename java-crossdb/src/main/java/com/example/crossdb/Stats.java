package com.example.crossdb;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/** 单次查询的执行统计：各源库实际下发的 SQL、拉取行数、Bind Join 批次与网络行数。
 *
 * <p>ponytail: 静态 {@link #ACTIVE} 供生成代码里的 BindJoinExec 读取（linq4j 常量
 * 无法携带任意对象）；一个 CrossDb 同一时刻只支持一条并发查询（CalciteConnection
 * 本身也不并发安全），多实例并发时统计会串场，需要按查询隔离时再改成 DataContext 传递。
 */
final class Stats {
  /** 当前正在执行的查询统计，由 CrossDb.query 在计划执行前设置。 */
  public static volatile Stats ACTIVE;

  final Map<String, Schema> schemas = new ConcurrentHashMap<>();
  final LongAdder bindBatches = new LongAdder();
  final LongAdder bindRows = new LongAdder();

  static final class Schema {
    final Set<String> sqls = ConcurrentHashMap.newKeySet();
    final LongAdder rows = new LongAdder();
  }

  void sql(String schema, String sql) {
    schema(schema).sqls.add(sql);
  }

  void rows(String schema, long n) {
    schema(schema).rows.add(n);
  }

  private Schema schema(String name) {
    return schemas.computeIfAbsent(name, k -> new Schema());
  }

  void batch(long keys, long fetchedRows) {
    bindBatches.increment();
    bindRows.add(fetchedRows);
  }

  void reset() {
    schemas.clear();
    bindBatches.reset();
    bindRows.reset();
  }

  String render() {
    StringBuilder sb = new StringBuilder("== crossdb 执行分析 ==\n");
    Map<String, Schema> sorted = new LinkedHashMap<>();
    new java.util.TreeMap<>(schemas).forEach(sorted::put);
    for (Map.Entry<String, Schema> e : sorted.entrySet()) {
      sb.append('[').append(e.getKey()).append("] networkRows=").append(e.getValue().rows.sum())
          .append('\n');
      for (String sql : e.getValue().sqls) {
        sb.append("  SQL: ").append(sql.replace('\n', ' ')).append('\n');
      }
    }
    sb.append("bindJoin: batches=").append(bindBatches.sum())
        .append(", bindRows=").append(bindRows.sum()).append('\n');
    return sb.toString();
  }
}
