package com.example.crossdb;

import org.apache.calcite.adapter.enumerable.EnumerableConvention;
import org.apache.calcite.adapter.enumerable.EnumerableLimit;
import org.apache.calcite.adapter.jdbc.JdbcConvention;
import org.apache.calcite.adapter.jdbc.JdbcRules.JdbcSort;
import org.apache.calcite.adapter.jdbc.JdbcToEnumerableConverter;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelCollation;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Sort;

import javax.sql.DataSource;

import java.util.Map;

/** Top-N 下推（ORDER BY + LIMIT 驱动侧裁剪）：
 * {@code LIMIT -> ORDER BY -> 跨库 Join} 且排序列全在驱动侧时，把 ORDER BY + OFFSET/FETCH
 * 一起下推进驱动侧源库 SQL（源库只返回 LIMIT 行），再走 Bind Join（内表按这批 key
 * 小 IN 查询），网络传输降为 O(LIMIT) 量级；本地保留 Sort/Limit 以约束最终输出
 * （join 可能放大行数，LIMIT 必须在 join 之后裁剪）。
 */
class TopNBindJoinRule extends RelOptRule {
  private final Map<String, DataSource> sources;
  private final int batchSize;
  private final int parallelism;

  TopNBindJoinRule(Map<String, DataSource> sources, int batchSize, int parallelism) {
    super(operand(EnumerableLimit.class,
        operand(Sort.class, operand(Join.class, any()))), "TopNBindJoinRule");
    this.sources = sources;
    this.batchSize = batchSize;
    this.parallelism = parallelism;
  }

  @Override public boolean matches(RelOptRuleCall call) {
    if (!(call.rel(0) instanceof EnumerableLimit limit)
        || limit.fetch == null
        || !(call.rel(1) instanceof Sort sort)
        || sort.getCollation().getFieldCollations().isEmpty()
        || !(call.rel(2) instanceof Join join)
        || join.getTraitSet().getConvention() != EnumerableConvention.INSTANCE
        || (join.getJoinType() != JoinRelType.INNER
            && join.getJoinType() != JoinRelType.LEFT)) {
      return false;
    }
    int leftCount = join.getLeft().getRowType().getFieldCount();
    return sort.getCollation().getFieldCollations().stream()
        .allMatch(fc -> fc.getFieldIndex() < leftCount);
  }

  @Override public void onMatch(RelOptRuleCall call) {
    try {
      attempt(call);
    } catch (Throwable e) {
      if (Boolean.getBoolean("crossdb.debug")) {
        System.err.println("TopNBindJoinRule: 生成候选失败");
        e.printStackTrace(System.err);
      }
    }
  }

  private void attempt(RelOptRuleCall call) {
    EnumerableLimit limit = call.rel(0);
    Sort sort = call.rel(1);
    Join join = call.rel(2);

    RelNode leftEnum = BindJoinRule.toEnumerable(join.getLeft());
    if (!(leftEnum instanceof JdbcToEnumerableConverter leftConverter)) {
      return;
    }
    RelNode leftJdbc = BindJoinRule.unwrapDeep(leftConverter.getInput());
    if (!(leftJdbc.getTraitSet().getConvention() instanceof JdbcConvention convention)) {
      return;
    }
    // ORDER BY + OFFSET/FETCH 一起下推进驱动侧源库 SQL
    RelCollation collation = sort.getCollation();
    RelTraitSet traits = leftJdbc.getCluster().traitSetOf(convention).replace(collation);
    JdbcSort pushed = new JdbcSort(leftJdbc.getCluster(), traits, leftJdbc,
        collation, limit.offset, limit.fetch);
    RelNode pushedEnum = new JdbcToEnumerableConverter(
        pushed.getCluster(),
        pushed.getCluster().traitSetOf(EnumerableConvention.INSTANCE), pushed) {
    };

    EnumerableBindJoin candidate = BindJoinRule.make(pushedEnum, join.getRight(),
        join.getCondition(), join.getJoinType(), sources, batchSize, parallelism);
    if (candidate == null) {
      return;
    }
    // 本地 Sort/Limit 保留：join 可能放大行数
    Sort localSort = sort.copy(sort.getTraitSet(), candidate,
        sort.getCollation(), null, null);
    call.transformTo(EnumerableLimit.create(localSort, limit.offset, limit.fetch));
    if (Boolean.getBoolean("crossdb.debug")) {
      System.err.println("TopNBindJoinRule: 已生成 Top-N 候选 fetch=" + limit.fetch);
    }
  }
}
