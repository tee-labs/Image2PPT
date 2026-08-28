package com.example.crossdb;

import org.apache.calcite.adapter.enumerable.EnumerableConvention;
import org.apache.calcite.adapter.jdbc.JdbcSchema;
import org.apache.calcite.config.Lex;
import org.apache.calcite.jdbc.CalciteConnection;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.Planner;
import org.apache.calcite.tools.Program;
import org.apache.calcite.tools.Programs;
import org.apache.calcite.tools.RelRunner;

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
 * 并发拉取，见 BindJoinRule）或本地 hash join。所有拉取带 fetchSize 流式读取与
 * 行数熔断（Guarded）。
 */
public class CrossDb implements AutoCloseable {
  public static final int DEFAULT_FETCH_SIZE = 1000;
  public static final long DEFAULT_ROW_LIMIT = 1_000_000L;
  public static final int DEFAULT_BIND_BATCH_SIZE = 500;
  public static final int DEFAULT_BIND_PARALLELISM = 4;

  private final CalciteConnection connection;
  private final Map<String, DataSource> sources = new HashMap<>();
  private final int fetchSize;
  private final long rowLimit;
  private final int bindBatchSize;
  private final int bindParallelism;

  public CrossDb() throws SQLException {
    this(DEFAULT_FETCH_SIZE, DEFAULT_ROW_LIMIT, DEFAULT_BIND_BATCH_SIZE,
        DEFAULT_BIND_PARALLELISM);
  }

  public CrossDb(int fetchSize, long rowLimit, int bindBatchSize, int bindParallelism)
      throws SQLException {
    if (fetchSize < 1 || rowLimit < 1 || bindBatchSize < 1 || bindParallelism < 1) {
      throw new IllegalArgumentException("fetchSize/rowLimit/batchSize/parallelism 必须 >= 1");
    }
    Properties info = new Properties();
    info.setProperty("lex", "MYSQL");
    Connection raw = DriverManager.getConnection("jdbc:calcite:", info);
    this.connection = raw.unwrap(CalciteConnection.class);
    this.fetchSize = fetchSize;
    this.rowLimit = rowLimit;
    this.bindBatchSize = bindBatchSize;
    this.bindParallelism = bindParallelism;
  }

  public CrossDb register(String schema, DataSource dataSource) throws SQLException {
    DataSource guarded = Guarded.wrap(dataSource, fetchSize, rowLimit);
    sources.put(schema, guarded);
    connection.getRootSchema().add(schema,
        JdbcSchema.create(connection.getRootSchema(), schema, guarded, null, null));
    return this;
  }

  public ResultSet query(String sql) throws SQLException {
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
        System.err.println(org.apache.calcite.plan.RelOptUtil.toString(best));
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
    ResultSet rs = connection.unwrap(RelRunner.class).prepareStatement(best).executeQuery();
    return Guarded.limit(rs, rowLimit);
  }

  private Program queryProgram() {
    BindJoinRule rule = new BindJoinRule(sources, bindBatchSize, bindParallelism);
    Program standard = Programs.standard();
    return (planner, rel, requiredTraits, materializations, lattices) -> {
      planner.addRule(rule);
      return standard.run(planner, rel, requiredTraits, materializations, lattices);
    };
  }

  @Override public void close() throws SQLException {
    connection.close();
  }
}
