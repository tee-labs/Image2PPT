package com.example.crossdb;

import org.apache.calcite.adapter.enumerable.EnumerableConvention;
import org.apache.calcite.adapter.jdbc.JdbcSchema;
import org.apache.calcite.adapter.jdbc.JdbcTableScan;
import org.apache.calcite.adapter.jdbc.JdbcToEnumerableConverter;
import org.apache.calcite.config.Lex;
import org.apache.calcite.jdbc.CalciteConnection;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.Sort;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.Planner;
import org.apache.calcite.tools.Program;
import org.apache.calcite.tools.Programs;

import javax.sql.DataSource;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

/** 引擎入口：register(schema, DataSource) 注册任意 JDBC 库，query(sql) 跨库执行。
 *
 * <p>同库过滤/JOIN/聚合自动下推方言 SQL；跨库 JOIN 走 Bind Join（分批 IN 下推 +
 * 并发拉取 + 流水线式流式执行，见 BindJoinRule/BindJoinExec）或本地 hash join。
 * 所有拉取带 fetchSize 流式读取、行数熔断与 queryTimeout 超时传播（Guarded）。
 *
 * <p>其他入口：
 * <ul>
 *   <li>{@link #explain} — 返回优化后的物理计划；</li>
 *   <li>{@link #analyze} — 执行并输出各源库实际下发的 SQL、网络行数、Bind Join 批次；</li>
 *   <li>{@link #safeMode()} — 拦截「无过滤条件的源库全表拉取」的查询（OLTP 防呆）。</li>
 * </ul>
 */
public class CrossDb implements AutoCloseable {
  public static final int DEFAULT_FETCH_SIZE = 1000;
  public static final long DEFAULT_ROW_LIMIT = 1_000_000L;
  public static final int DEFAULT_BIND_BATCH_SIZE = 500;
  public static final int DEFAULT_BIND_PARALLELISM = 4;
  public static final int DEFAULT_QUERY_TIMEOUT = 0;

  private final CalciteConnection connection;
  private final Map<String, DataSource> sources = new HashMap<>();
  private final Stats stats = new Stats();
  private final int fetchSize;
  private final long rowLimit;
  private final int bindBatchSize;
  private final int bindParallelism;
  private final int queryTimeout;
  private boolean safeMode;

  public CrossDb() throws SQLException {
    this(DEFAULT_FETCH_SIZE, DEFAULT_ROW_LIMIT, DEFAULT_BIND_BATCH_SIZE,
        DEFAULT_BIND_PARALLELISM, DEFAULT_QUERY_TIMEOUT);
  }

  public CrossDb(int fetchSize, long rowLimit, int bindBatchSize, int bindParallelism)
      throws SQLException {
    this(fetchSize, rowLimit, bindBatchSize, bindParallelism, DEFAULT_QUERY_TIMEOUT);
  }

  /** @param queryTimeout 每条源库语句的 queryTimeout 秒数（0=不限制），超时由 JDBC
   * 驱动在源库侧取消执行，防止慢查询拖垮连接池 */
  public CrossDb(int fetchSize, long rowLimit, int bindBatchSize, int bindParallelism,
      int queryTimeout) throws SQLException {
    if (fetchSize < 1 || rowLimit < 1 || bindBatchSize < 1 || bindParallelism < 1
        || queryTimeout < 0) {
      throw new IllegalArgumentException("fetchSize/rowLimit/batchSize/parallelism 必须 >= 1，"
          + "queryTimeout 必须 >= 0");
    }
    Properties info = new Properties();
    info.setProperty("lex", "MYSQL");
    Connection raw = DriverManager.getConnection("jdbc:calcite:", info);
    this.connection = raw.unwrap(CalciteConnection.class);
    this.fetchSize = fetchSize;
    this.rowLimit = rowLimit;
    this.bindBatchSize = bindBatchSize;
    this.bindParallelism = bindParallelism;
    this.queryTimeout = queryTimeout;
  }

  /** 开启 safeMode：计划中出现「无过滤条件的源库全表拉取」（Bind Join 内表除外，
   * 其始终带 key IN 过滤；聚合视为有归约）时拒绝执行。 */
  public CrossDb safeMode() {
    this.safeMode = true;
    return this;
  }

  public CrossDb register(String schema, DataSource dataSource) throws SQLException {
    DataSource guarded =
        Guarded.wrap(dataSource, fetchSize, rowLimit, schema, stats, queryTimeout);
    sources.put(schema, guarded);
    connection.getRootSchema().add(schema,
        JdbcSchema.create(connection.getRootSchema(), schema, guarded, null, null));
    return this;
  }

  public ResultSet query(String sql) throws SQLException {
    RelNode best = plan(sql);
    if (safeMode) {
      checkSafe(best);
    }
    // Stats.ACTIVE 保留到结果集迭代结束（Bind Join 在迭代期执行）；
    // 同一 CrossDb 并发多条查询时统计会串场，CalciteConnection 本身也不支持并发。
    stats.reset();
    Stats.ACTIVE = stats;
    ResultSet rs = Exec.query(connection, best);
    return Guarded.limit(rs, rowLimit);
  }

  /** 返回优化后的物理计划（EXPLAIN）。 */
  public String explain(String sql) throws SQLException {
    return RelOptUtil.toString(plan(sql));
  }

  /** EXPLAIN ANALYZE：执行查询并返回物理计划 + 各源库实际下发的 SQL、
   * 每库网络行数、Bind Join 批次与拉取行数。注意会消费整个结果集。 */
  public String analyze(String sql) throws SQLException {
    RelNode best = plan(sql);
    if (safeMode) {
      checkSafe(best);
    }
    stats.reset();
    Stats.ACTIVE = stats;
    try (ResultSet rs = Guarded.limit(Exec.query(connection, best), rowLimit)) {
      while (rs.next()) {
        // 全量消费以统计网络行数
      }
    } finally {
      Stats.ACTIVE = null;
    }
    return RelOptUtil.toString(best) + "\n" + stats.render();
  }

  private RelNode plan(String sql) throws SQLException {
    FrameworkConfig config = Frameworks.newConfigBuilder()
        .parserConfig(SqlParser.config().withLex(Lex.MYSQL))
        .defaultSchema(connection.getRootSchema())
        .programs(queryProgram())
        .build();
    Planner planner = Frameworks.getPlanner(config);
    RelNode best;
    try {
      SqlNode parsed = planner.parse(sql);
      SqlNode validated = planner.validate(parsed);
      RelRoot root = planner.rel(validated);
      best = planner.transform(0,
          root.rel.getTraitSet().replace(EnumerableConvention.INSTANCE), root.rel);
      if (Boolean.getBoolean("crossdb.debug")) {
        System.err.println(RelOptUtil.toString(best));
      }
    } catch (Exception e) {
      if (Boolean.getBoolean("crossdb.debug")) {
        e.printStackTrace(System.err);
        for (Throwable c : e.getSuppressed()) {
          c.printStackTrace(System.err);
        }
      }
      throw e instanceof SQLException se ? se : new SQLException(e);
    } finally {
      planner.close();
    }
    return best;
  }

  /** 防呆：源库拉取子树不允许只有 Scan/Filter/Project/无 LIMIT Sort 链（全表拉取）。
   * Bind Join 内表例外——其 SQL 运行时必带 key IN 过滤；Aggregate 视为有归约。 */
  private void checkSafe(RelNode plan) throws CrossDbUnsafeQueryException {
    RelNode n = BindJoinRule.unwrapSubset(plan);
    if (n instanceof EnumerableBindJoin bind) {
      checkSafe(BindJoinRule.unwrapSubset(bind.getLeft()));
      return;
    }
    if (n instanceof JdbcToEnumerableConverter converter
        && isBarePull(BindJoinRule.unwrapDeep(converter.getInput()))) {
      throw new CrossDbUnsafeQueryException("crossdb safeMode: 检测到无过滤条件的源库"
          + "全表拉取（" + BindJoinRule.unwrapDeep(converter.getInput()).explain()
          + "），已拒绝执行。请补充 WHERE 条件或 LIMIT。");
    }
    for (RelNode input : n.getInputs()) {
      checkSafe(input);
    }
  }

  private static boolean isBarePull(RelNode n) {
    while (true) {
      if (n instanceof JdbcTableScan) {
        return true;
      }
      // 有行数归约能力的算子视作已过滤
      if (n instanceof org.apache.calcite.rel.core.Filter
          || n instanceof org.apache.calcite.rel.core.Aggregate) {
        return false;
      }
      if (n instanceof Sort s) {
        if (s.offset == null && s.fetch == null && s.getInputs().size() == 1) {
          n = s.getInput(0);
          continue;
        }
        return false;
      }
      if (n.getInputs().size() == 1 && !(n instanceof Join)) {
        n = n.getInput(0);
        continue;
      }
      return false;
    }
  }

  private Program queryProgram() {
    BindJoinRule rule = new BindJoinRule(sources, bindBatchSize, bindParallelism);
    TopNBindJoinRule topN = new TopNBindJoinRule(sources, bindBatchSize, bindParallelism);
    Program standard = Programs.standard();
    return (planner, rel, requiredTraits, materializations, lattices) -> {
      planner.addRule(rule);
      planner.addRule(topN);
      return standard.run(planner, rel, requiredTraits, materializations, lattices);
    };
  }

  @Override public void close() throws SQLException {
    connection.close();
  }
}
