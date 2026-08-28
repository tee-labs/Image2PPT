package com.example.crossdb;

import org.apache.calcite.adapter.enumerable.EnumerableConvention;
import org.apache.calcite.adapter.jdbc.JdbcConvention;
import org.apache.calcite.adapter.jdbc.JdbcImplementor;
import org.apache.calcite.adapter.jdbc.JdbcTableScan;
import org.apache.calcite.adapter.jdbc.JdbcToEnumerableConverter;
import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlKind;

import javax.sql.DataSource;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

/** 跨库 INNER JOIN 的 Bind Join 规则：驱动侧 key 分批 IN 下推到内表库并发拉取，
 * 替代「内表全量拉取 + 本地 hash join」。
 *
 * <p>在 Enumerable 阶段的 Join 上触发（子节点已是具体 Enumerable 节点）。
 * 仅在右子树可整体下推为一条 JDBC SQL、连接条件为单等值对、左右来自不同库时触发；
 * 其余情况维持 Calcite 原生计划。
 */
class BindJoinRule extends RelOptRule {
  private final Map<String, DataSource> sources;
  private final int batchSize;
  private final int parallelism;

  BindJoinRule(Map<String, DataSource> sources, int batchSize, int parallelism) {
    super(operand(Join.class, any()), "BindJoinRule");
    this.sources = sources;
    this.batchSize = batchSize;
    this.parallelism = parallelism;
  }

  @Override public boolean matches(RelOptRuleCall call) {
    return call.rel(0) instanceof Join join
        && join.getTraitSet().getConvention() == EnumerableConvention.INSTANCE
        && join.getJoinType() == JoinRelType.INNER;
  }

  @Override public void onMatch(RelOptRuleCall call) {
    try {
      attempt(call);
    } catch (Throwable e) {
      if (Boolean.getBoolean("crossdb.debug")) {
        System.err.println("BindJoinRule: 生成候选失败");
        e.printStackTrace(System.err);
      }
    }
  }

  private void attempt(RelOptRuleCall call) {
    Join join = call.rel(0);
    if (Boolean.getBoolean("crossdb.debug")) {
      System.err.println("BindJoinRule: Enumerable Join 命中 " + join.getClass().getSimpleName()
          + " right=" + join.getRight().getClass().getSimpleName());
    }
    int leftCount = join.getLeft().getRowType().getFieldCount();
    List<RexNode> conjuncts = org.apache.calcite.plan.RelOptUtil.conjunctions(join.getCondition());
    if (conjuncts.size() != 1
        || !(conjuncts.get(0) instanceof RexCall eq)
        || eq.getKind() != SqlKind.EQUALS
        || eq.getOperands().size() != 2
        || !(eq.getOperands().get(0) instanceof RexInputRef a)
        || !(eq.getOperands().get(1) instanceof RexInputRef b)) {
      return;
    }
    int leftKey;
    int rightKey;
    if (a.getIndex() < leftCount && b.getIndex() >= leftCount) {
      leftKey = a.getIndex();
      rightKey = b.getIndex() - leftCount;
    } else if (b.getIndex() < leftCount && a.getIndex() >= leftCount) {
      leftKey = b.getIndex();
      rightKey = a.getIndex() - leftCount;
    } else {
      return;
    }
    if (!a.getType().getSqlTypeName().equals(b.getType().getSqlTypeName())) {
      return;
    }

    RelNode leftEnum = toEnumerable(join.getLeft());
    RelNode rightEnum = toEnumerable(join.getRight());
    if (!(rightEnum instanceof JdbcToEnumerableConverter converter)) {
      return;
    }
    JdbcTableScan rightScan = underlyingScan(converter.getInput());
    if (rightScan == null || !(rightScan.getConvention() instanceof JdbcConvention convention)) {
      return;
    }
    JdbcTableScan leftScan = firstJdbcScan(leftEnum);
    if (leftScan == null
        || leftScan.getConvention() == convention
        || schemaOf(leftScan).equals(schemaOf(rightScan))) {
      return;
    }
    DataSource dataSource = sources.get(schemaOf(rightScan));
    if (dataSource == null) {
      return;
    }
    List<String> rightNames = rightEnum.getRowType().getFieldNames();
    if (rightNames.size() != new HashSet<>(rightNames).size()) {
      return;
    }

    String innerSql = new JdbcImplementor(convention.dialect,
        (JavaTypeFactory) join.getCluster().getTypeFactory())
        .visitRoot(converter.getInput()).asSelect().toSqlString(convention.dialect).getSql();
    String t = convention.dialect.quoteIdentifier("T");
    String sqlPrefix = "SELECT * FROM (" + innerSql + ") AS " + t;
    String keyRef = t + "." + convention.dialect.quoteIdentifier(rightNames.get(rightKey));

    call.transformTo(EnumerableBindJoin.create(leftEnum, rightEnum, join.getCondition(),
        schemaOf(rightScan), sqlPrefix, keyRef, leftKey, rightKey, rightNames.size(), batchSize,
        parallelism));
    if (Boolean.getBoolean("crossdb.debug")) {
      System.err.println("BindJoinRule: 已生成候选计划 innerSql=" + sqlPrefix);
    }
  }

  /** 把匹配到的子节点解析成「具体」的 Enumerable 节点：
   * subset 展开为 best/original，仍是 JDBC/逻辑子树时包一层转换器。 */
  private static RelNode toEnumerable(RelNode n) {
    n = unwrapDeep(n);
    if (n.getConvention() == EnumerableConvention.INSTANCE) {
      return n;
    }
    return new JdbcToEnumerableConverter(
        n.getCluster(), n.getTraitSet().replace(EnumerableConvention.INSTANCE), n) {
    };
  }

  private static JdbcTableScan underlyingScan(RelNode n) {
    while (true) {
      if (n instanceof JdbcTableScan s) {
        return s;
      }
      if (n.getInputs().size() == 1) {
        n = n.getInput(0);
      } else {
        return null;
      }
    }
  }

  /** SQL 生成用：把子树里的 RelSubset 全部换成 best/original 具体节点。 */
  private static RelNode unwrapDeep(RelNode n) {
    n = unwrapSubset(n);
    List<RelNode> inputs = new ArrayList<>();
    boolean changed = false;
    for (RelNode in : n.getInputs()) {
      RelNode u = unwrapDeep(in);
      changed |= u != in;
      inputs.add(u);
    }
    return changed ? n.copy(n.getTraitSet(), inputs) : n;
  }

  private static RelNode unwrapSubset(RelNode n) {
    while (n instanceof org.apache.calcite.plan.volcano.RelSubset s) {
      n = s.getBestOrOriginal();
    }
    return n;
  }

  private static JdbcTableScan firstJdbcScan(RelNode n) {
    List<JdbcTableScan> found = new ArrayList<>();
    collectJdbcScans(n, found, 0);
    return found.isEmpty() ? null : found.get(0);
  }

  private static void collectJdbcScans(RelNode n, List<JdbcTableScan> out, int depth) {
    if (depth > 10) {
      return;
    }
    if (n instanceof JdbcTableScan s) {
      out.add(s);
      return;
    }
    for (RelNode child : n.getInputs()) {
      collectJdbcScans(child, out, depth + 1);
    }
  }

  private static String schemaOf(JdbcTableScan scan) {
    List<String> names = scan.getTable().getQualifiedName();
    return names.get(names.size() - 2);
  }
}
