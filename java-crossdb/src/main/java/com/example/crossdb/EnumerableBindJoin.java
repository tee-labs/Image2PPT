package com.example.crossdb;

import org.apache.calcite.adapter.enumerable.EnumerableConvention;
import org.apache.calcite.adapter.enumerable.EnumerableRel;
import org.apache.calcite.adapter.enumerable.EnumerableRelImplementor;
import org.apache.calcite.adapter.enumerable.JavaRowFormat;
import org.apache.calcite.adapter.enumerable.PhysType;
import org.apache.calcite.adapter.enumerable.PhysTypeImpl;
import org.apache.calcite.linq4j.function.Function1;
import org.apache.calcite.linq4j.tree.BlockBuilder;
import org.apache.calcite.linq4j.tree.Expression;
import org.apache.calcite.linq4j.tree.Expressions;
import org.apache.calcite.linq4j.tree.ParameterExpression;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptCost;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelWriter;
import org.apache.calcite.rel.core.CorrelationId;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.util.BuiltInMethod;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

import javax.sql.DataSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/** Bind Join 物理算子：左侧 Enumerable 作为驱动侧流式读取，右侧 JDBC 子树的 SQL
 * 以 {@code SELECT * FROM (inner) T WHERE keys IN (批内 key)} 形式分批并发执行。
 * 支持 INNER 与 LEFT（LEFT 时未匹配的外表行右侧补 NULL）。
 */
class EnumerableBindJoin extends Join implements EnumerableRel {
  final String schemaName;
  final String sqlPrefix;
  final String[] keyCols;
  final int[] leftKeys;
  final int[] rightKeys;
  final int rightFieldCount;
  final int batchSize;
  final int parallelism;
  final boolean tupleIn;

  private EnumerableBindJoin(RelOptCluster cluster, RelTraitSet traits, RelNode left,
      RelNode right, RexNode condition, Set<CorrelationId> variablesSet, String schemaName,
      String sqlPrefix, String[] keyCols, int[] leftKeys, int[] rightKeys, int rightFieldCount,
      int batchSize, int parallelism, boolean tupleIn, JoinRelType joinType) {
    super(cluster, traits, ImmutableList.of(), left, right, condition, variablesSet, joinType);
    this.schemaName = schemaName;
    this.sqlPrefix = sqlPrefix;
    this.keyCols = keyCols;
    this.leftKeys = leftKeys;
    this.rightKeys = rightKeys;
    this.rightFieldCount = rightFieldCount;
    this.batchSize = batchSize;
    this.parallelism = parallelism;
    this.tupleIn = tupleIn;
  }

  static EnumerableBindJoin create(RelNode left, RelNode right, RexNode condition,
      String schemaName, String sqlPrefix, String[] keyCols, int[] leftKeys, int[] rightKeys,
      int rightFieldCount, int batchSize, int parallelism, boolean tupleIn,
      JoinRelType joinType) {
    RelOptCluster cluster = left.getCluster();
    return new EnumerableBindJoin(cluster,
        cluster.traitSetOf(EnumerableConvention.INSTANCE),
        left, right, condition, ImmutableSet.of(), schemaName, sqlPrefix, keyCols, leftKeys,
        rightKeys, rightFieldCount, batchSize, parallelism, tupleIn, joinType);
  }

  @Override public EnumerableBindJoin copy(RelTraitSet traitSet, RexNode condition, RelNode left,
      RelNode right, JoinRelType joinType, boolean semiJoinDone) {
    if (joinType != JoinRelType.INNER && joinType != JoinRelType.LEFT) {
      throw new AssertionError("EnumerableBindJoin 不支持 " + joinType);
    }
    return new EnumerableBindJoin(getCluster(), traitSet, left, right, condition, variablesSet,
        schemaName, sqlPrefix, keyCols, leftKeys, rightKeys, rightFieldCount, batchSize,
        parallelism, tupleIn, joinType);
  }

  @Override public RelWriter explainTerms(RelWriter pw) {
    return super.explainTerms(pw)
        .item("joinType", getJoinType())
        .item("keys", keyCols.length)
        .item("tupleIn", tupleIn)
        .item("batchSize", batchSize)
        .item("parallelism", parallelism);
  }

  @Override public RelOptCost computeSelfCost(RelOptPlanner planner, RelMetadataQuery mq) {
    double rowCount = mq.getRowCount(this);
    double leftRowCount = mq.getRowCount(getLeft());
    double rightRowCount = mq.getRowCount(getRight());
    if (Double.isInfinite(rowCount) || Double.isInfinite(leftRowCount)
        || Double.isInfinite(rightRowCount)) {
      return planner.getCostFactory().makeInfiniteCost();
    }
    RelOptCost rightCost = planner.getCost(getRight(), mq);
    if (rightCost == null) {
      return null;
    }
    double batches = Math.max(1.0, Math.ceil(leftRowCount / batchSize));
    // 内表网络行数按权重计入 IO 成本：内表被谓词（过滤/传递谓词）压得越狠越优先
    double io = rightRowCount * 0.1;
    return planner.getCostFactory()
        .makeCost(rowCount + leftRowCount + io, 0, io)
        .plus(rightCost.multiplyBy(batches));
  }

  @Override public EnumerableRel.Result implement(EnumerableRelImplementor implementor,
      EnumerableRel.Prefer pref) {
    try {
      return doImplement(implementor, pref);
    } catch (RuntimeException e) {
      if (Boolean.getBoolean("crossdb.debug")) {
        System.err.println("EnumerableBindJoin.implement 失败:");
        e.printStackTrace(System.err);
      }
      throw e;
    }
  }

  private EnumerableRel.Result doImplement(EnumerableRelImplementor implementor,
      EnumerableRel.Prefer pref) {
    final BlockBuilder builder = new BlockBuilder();
    final EnumerableRel.Result leftResult =
        implementor.visitChild(this, 0, (EnumerableRel) getLeft(), pref);
    final Expression leftE = builder.append("left", leftResult.block);
    final ParameterExpression lrow =
        Expressions.parameter(leftResult.physType.getJavaRowType(), "lrow");
    final BlockBuilder mapperBlock = new BlockBuilder();
    mapperBlock.add(Expressions.return_(null,
        leftResult.physType.convertTo(lrow, JavaRowFormat.ARRAY)));
    final Expression mapper = Expressions.lambda(Function1.class, mapperBlock.toBlock(), lrow);
    final Expression leftRows = builder.append("leftRows",
        Expressions.call(leftE, BuiltInMethod.SELECT.method, mapper));
    final Expression rootSchema = Expressions.call(
        org.apache.calcite.DataContext.ROOT,
        BuiltInMethod.DATA_CONTEXT_GET_ROOT_SCHEMA.method);
    final Expression subSchema = Expressions.call(
        rootSchema, BuiltInMethod.SCHEMA_GET_SUB_SCHEMA.method,
        Expressions.constant(schemaName));
    builder.add(Expressions.call(BindJoinExec.class, "join",
        leftRows,
        org.apache.calcite.schema.Schemas.unwrap(subSchema, DataSource.class),
        Expressions.constant(sqlPrefix),
        constantArray(String.class, keyCols),
        constantArray(int.class, leftKeys),
        constantArray(int.class, rightKeys),
        Expressions.constant(rightFieldCount),
        Expressions.constant(batchSize),
        Expressions.constant(parallelism),
        Expressions.constant(getJoinType() == JoinRelType.LEFT),
        Expressions.constant(tupleIn)));
    final PhysType physType =
        PhysTypeImpl.of(implementor.getTypeFactory(), getRowType(), JavaRowFormat.ARRAY);
    return implementor.result(physType, builder.toBlock());
  }

  private static Expression constantArray(Class<?> type, int[] values) {
    List<Expression> items = new ArrayList<>();
    for (int v : values) {
      items.add(Expressions.constant(v));
    }
    return Expressions.newArrayInit(type, items);
  }

  private static Expression constantArray(Class<?> type, String[] values) {
    List<Expression> items = new ArrayList<>();
    for (String v : values) {
      items.add(Expressions.constant(v));
    }
    return Expressions.newArrayInit(type, items);
  }
}
