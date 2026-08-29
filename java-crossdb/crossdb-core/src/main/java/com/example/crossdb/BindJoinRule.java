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
import org.apache.calcite.sql.SqlDialect;
import org.apache.calcite.sql.SqlKind;

import javax.sql.DataSource;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** 跨库 JOIN 的 Bind Join 规则：驱动侧 key 分批 IN 下推到内表库并发拉取，
 * 替代「内表全量拉取 + 本地 hash join」。
 *
 * <p>在 Enumerable 阶段的 Join 上触发（子节点已是具体 Enumerable 节点）。
 * 支持 INNER/LEFT/RIGHT（RIGHT 交换内外侧）/FULL（外表耗尽后 NOT IN 反连接补内表）/
 * SEMI/ANTI（EXISTS/NOT EXISTS 去相关后的半连接，仅输出外表行）；
 * 连接条件须全部为跨侧等值对（支持多列复合键，无残余条件），
 * 内表可整体下推为一条 JDBC SQL，左右来自不同库；其余情况维持 Calcite 原生计划。
 * 多列 key 在 H2/MySQL/PostgreSQL/Oracle 用 tuple-IN，其余方言降级为 OR 组。
 */
class BindJoinRule extends RelOptRule {
  private static final Set<SqlDialect.DatabaseProduct> TUPLE_IN_DIALECTS = Set.of(
      SqlDialect.DatabaseProduct.H2,
      SqlDialect.DatabaseProduct.MYSQL,
      SqlDialect.DatabaseProduct.POSTGRESQL,
      SqlDialect.DatabaseProduct.ORACLE);

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
        && join.getTraitSet().getConvention() == EnumerableConvention.INSTANCE;
  }

  @Override public void onMatch(RelOptRuleCall call) {
    try {
      Join join = call.rel(0);
      RelNode drive = join.getLeft();
      RelNode inner = join.getRight();
      RexNode condition = join.getCondition();
      if (join.getJoinType() == JoinRelType.RIGHT) {
        // RIGHT 交换内外侧按 LEFT 形态执行；条件列引用从 [左 ++ 右] 重映射到 [右 ++ 左]，
        // 行型由 EnumerableBindJoin.deriveJoinRowType 恢复为原始 [左 ++ 右]
        drive = join.getRight();
        inner = join.getLeft();
        condition = swapCondition(join.getCluster().getRexBuilder(), condition,
            join.getLeft().getRowType().getFieldCount(),
            join.getRight().getRowType().getFieldCount());
      }
      EnumerableBindJoin candidate =
          make(drive, inner, condition, join.getJoinType(), sources, batchSize, parallelism);
      if (candidate != null) {
        call.transformTo(candidate);
        if (Boolean.getBoolean("crossdb.debug")) {
          System.err.println("BindJoinRule: 已生成候选计划 innerSql=" + candidate.sqlPrefix);
        }
      } else if (Boolean.getBoolean("crossdb.debug")) {
        System.err.println("BindJoinRule: null 候选 type=" + join.getJoinType()
            + " cond=" + condition);
      }
    } catch (Throwable e) {
      if (Boolean.getBoolean("crossdb.debug")) {
        System.err.println("BindJoinRule: 生成候选失败");
        e.printStackTrace(System.err);
      }
    }
  }

  /** RIGHT 交换后列引用重映射：原左表第 i 列 → 右表宽度 + i；原右表第 j 列 → j。 */
  static RexNode swapCondition(org.apache.calcite.rex.RexBuilder rexBuilder, RexNode condition,
      int leftCount, int rightCount) {
    return condition.accept(new org.apache.calcite.rex.RexShuttle() {
      @Override public RexNode visitInputRef(RexInputRef ref) {
        int i = ref.getIndex();
        return new RexInputRef(i < leftCount ? i + rightCount : i - leftCount, ref.getType());
      }
    });
  }

  /** 判定 (left, right) 是否可 Bind Join，可以则返回算子，否则 null。
   * left/right 为 Join 的原始输入。供本规则与 TopNBindJoinRule 复用。 */
  static EnumerableBindJoin make(RelNode leftRaw, RelNode rightRaw, RexNode condition,
      JoinRelType joinType, Map<String, DataSource> sources, int batchSize, int parallelism) {
    RelNode leftEnum = toEnumerable(leftRaw);
    RelNode rightEnum = toEnumerable(rightRaw);
    int leftCount = leftEnum.getRowType().getFieldCount();

    List<RexNode> conjuncts = org.apache.calcite.plan.RelOptUtil.conjunctions(condition);
    Map<Integer, Integer> pairs = new LinkedHashMap<>();
    for (RexNode conjunct : conjuncts) {
      // 去相关 SEMI/ANTI 自带「右 key IS NOT NULL」conjunct：哈希探测的 key 永不含
      // NULL（keyOf 返回 null 即跳过），等值匹配语义一致，直接忽略
      if ((joinType == JoinRelType.SEMI || joinType == JoinRelType.ANTI)
          && conjunct instanceof RexCall isnn
          && isnn.getKind() == SqlKind.IS_NOT_NULL
          && isnn.getOperands().get(0) instanceof RexInputRef) {
        continue;
      }
      if (!(conjunct instanceof RexCall eq)
          || eq.getKind() != SqlKind.EQUALS
          || eq.getOperands().size() != 2
          || !(eq.getOperands().get(0) instanceof RexInputRef a)
          || !(eq.getOperands().get(1) instanceof RexInputRef b)) {
        return null;
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
        return null;
      }
      if (!a.getType().getSqlTypeName().equals(b.getType().getSqlTypeName())) {
        return null;
      }
      Integer prev = pairs.put(leftKey, rightKey);
      if (prev != null && prev != rightKey) {
        return null;
      }
    }
    int[] leftKeys = pairs.keySet().stream().mapToInt(Integer::intValue).toArray();
    int[] rightKeys = pairs.values().stream().mapToInt(Integer::intValue).toArray();

    if (!(rightEnum instanceof JdbcToEnumerableConverter converter)) {
      return null;
    }
    JdbcTableScan rightScan = underlyingScan(converter.getInput());
    if (rightScan == null || !(rightScan.getConvention() instanceof JdbcConvention convention)) {
      return null;
    }
    JdbcTableScan leftScan = firstJdbcScan(leftEnum);
    if (leftScan == null
        || leftScan.getConvention() == convention
        || schemaOf(leftScan).equals(schemaOf(rightScan))) {
      return null;
    }
    DataSource dataSource = sources.get(schemaOf(rightScan));
    if (dataSource == null) {
      return null;
    }
    List<String> rightNames = rightEnum.getRowType().getFieldNames();
    if (rightNames.size() != new HashSet<>(rightNames).size()) {
      return null;
    }

    // 传递谓词下推：驱动侧 key 上的常量条件沿 join key 推到内表侧，
    // 使 `ON a.t=b.t` + `WHERE a.t=...` 在内表源库也生效，从源头减少网络传输。
    org.apache.calcite.rex.RexBuilder rexBuilder = leftEnum.getCluster().getRexBuilder();
    RelNode rightJdbcInput = converter.getInput();
    List<RexNode> innerInferred =
        keyConstantFilters(leftEnum, leftKeys, rightEnum, rightKeys, rexBuilder);
    // 去重：内表侧已有的等值过滤（按列名+常量值比较）不再重复包裹
    if (!innerInferred.isEmpty()) {
      Set<String> existing = new HashSet<>();
      RelNode n = unwrapSubset(rightJdbcInput);
      while (true) {
        if (n instanceof org.apache.calcite.rel.core.Filter f) {
          for (RexNode c : org.apache.calcite.plan.RelOptUtil.conjunctions(f.getCondition())) {
            if (c instanceof RexCall eq && eq.getKind() == SqlKind.EQUALS
                && eq.getOperands().size() == 2
                && eq.getOperands().get(0) instanceof RexInputRef ref
                && eq.getOperands().get(1) instanceof org.apache.calcite.rex.RexLiteral lit) {
              existing.add(f.getInput().getRowType().getFieldNames().get(ref.getIndex())
                  + "=" + lit.getValue2());
            }
          }
          n = unwrapSubset(f.getInput());
        } else if (n instanceof JdbcToEnumerableConverter c) {
          n = unwrapSubset(c.getInput());
        } else if (n instanceof org.apache.calcite.rel.core.Project) {
          // 过滤可能被 Calcite 下推到 Project 之下，按列名比较所以直接下探
          n = unwrapSubset(n.getInput(0));
        } else {
          break;
        }
      }
      List<String> rightNames2 = rightEnum.getRowType().getFieldNames();
      innerInferred.removeIf(r -> {
        org.apache.calcite.rex.RexCall eq = (org.apache.calcite.rex.RexCall) r;
        int idx = ((RexInputRef) eq.getOperands().get(0)).getIndex();
        Object v = ((org.apache.calcite.rex.RexLiteral) eq.getOperands().get(1)).getValue2();
        return !existing.add(rightNames2.get(idx) + "=" + v);
      });
    }
    if (!innerInferred.isEmpty()) {
      RelNode filtered = new org.apache.calcite.adapter.jdbc.JdbcRules.JdbcFilter(
          rightJdbcInput.getCluster(), rightJdbcInput.getTraitSet(), rightJdbcInput,
          org.apache.calcite.rex.RexUtil.composeConjunction(rexBuilder, innerInferred));
      rightJdbcInput = filtered;
      // 右子节点也带上过滤，使行数估计与代价正确反映内表的实际拉取量
      rightEnum = new JdbcToEnumerableConverter(
          filtered.getCluster(),
          filtered.getTraitSet().replace(EnumerableConvention.INSTANCE), filtered) {
      };
    }

    String innerSql = new JdbcImplementor(convention.dialect,
        (JavaTypeFactory) leftEnum.getCluster().getTypeFactory())
        .visitRoot(rightJdbcInput).asSelect().toSqlString(convention.dialect).getSql();
    String t = convention.dialect.quoteIdentifier("T");
    String sqlPrefix = "SELECT * FROM (" + innerSql + ") AS " + t;
    String[] keyCols = java.util.Arrays.stream(rightKeys)
        .mapToObj(i -> t + "." + convention.dialect.quoteIdentifier(rightNames.get(i)))
        .toArray(String[]::new);
    boolean tupleIn = TUPLE_IN_DIALECTS.contains(convention.dialect.getDatabaseProduct());

    return EnumerableBindJoin.create(leftEnum, rightEnum, condition, schemaOf(rightScan),
        sqlPrefix, keyCols, leftKeys, rightKeys, rightNames.size(), batchSize, parallelism,
        tupleIn, driverSortedByKey(leftEnum, leftKeys), joinType);
  }

  /** 驱动侧是否按 join key 有序（collation 前缀恰为 leftKeys）——此时 Bind Join 可做
   * 每窗口哈希淘汰。只认「子集/转换器之下顶层就是 Sort」的形态，宁可漏判不可误判。 */
  static boolean driverSortedByKey(RelNode leftEnum, int[] leftKeys) {
    RelNode n = unwrapSubset(leftEnum);
    while (n instanceof JdbcToEnumerableConverter c) {
      n = unwrapSubset(c.getInput());
    }
    if (!(n instanceof org.apache.calcite.rel.core.Sort s)) {
      return false;
    }
    List<org.apache.calcite.rel.RelFieldCollation> fcs =
        s.getCollation().getFieldCollations();
    if (fcs.size() < leftKeys.length) {
      return false;
    }
    for (int i = 0; i < leftKeys.length; i++) {
      if (fcs.get(i).getFieldIndex() != leftKeys[i]) {
        return false;
      }
    }
    return true;
  }

  /** 收集 source 侧 join key 上的常量等值条件（{@code key = 'v'}），翻译成 target 侧的
   * 等价条件。沿连续 Filter 链下探（索引与 source 顶部行型对齐），可穿过转换器，
   * 遇 Project 即停。 */
  private static List<RexNode> keyConstantFilters(RelNode source, int[] sourceKeys,
      RelNode target, int[] targetKeys, org.apache.calcite.rex.RexBuilder rexBuilder) {
    List<RexNode> inferred = new ArrayList<>();
    RelNode n = unwrapSubset(source);
    n = unwrapConverter(n);
    while (n instanceof org.apache.calcite.rel.core.Filter f) {
      for (RexNode conjunct : org.apache.calcite.plan.RelOptUtil.conjunctions(f.getCondition())) {
        if (!(conjunct instanceof RexCall eq)
            || eq.getKind() != SqlKind.EQUALS
            || eq.getOperands().size() != 2) {
          continue;
        }
        RexNode a = eq.getOperands().get(0);
        RexNode b = eq.getOperands().get(1);
        RexInputRef ref = a instanceof RexInputRef r ? r
            : (b instanceof RexInputRef r2 ? r2 : null);
        RexNode literal = a instanceof RexInputRef ? b : a;
        if (ref == null || !(literal instanceof org.apache.calcite.rex.RexLiteral)) {
          continue;
        }
        for (int k = 0; k < sourceKeys.length; k++) {
          if (sourceKeys[k] == ref.getIndex()) {
            int targetIdx = targetKeys[k];
            org.apache.calcite.rel.type.RelDataTypeField field =
                target.getRowType().getFieldList().get(targetIdx);
            inferred.add(rexBuilder.makeCall(org.apache.calcite.sql.fun.SqlStdOperatorTable.EQUALS,
                new RexInputRef(targetIdx, field.getType()), literal));
          }
        }
      }
      n = unwrapConverter(unwrapSubset(f.getInput()));
    }
    return inferred;
  }


  private static RelNode unwrapConverter(RelNode n) {
    return n instanceof JdbcToEnumerableConverter c ? unwrapSubset(c.getInput()) : n;
  }


  /** 把匹配到的子节点解析成「具体」的 Enumerable 节点：
   * subset 展开为 best/original，仍是 JDBC/逻辑子树时包一层转换器。 */
  static RelNode toEnumerable(RelNode n) {
    n = unwrapDeep(n);
    if (n.getConvention() == EnumerableConvention.INSTANCE) {
      return n;
    }
    return new JdbcToEnumerableConverter(
        n.getCluster(), n.getTraitSet().replace(EnumerableConvention.INSTANCE), n) {
    };
  }

  static JdbcTableScan underlyingScan(RelNode n) {
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
  static RelNode unwrapDeep(RelNode n) {
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

  static RelNode unwrapSubset(RelNode n) {
    while (n instanceof org.apache.calcite.plan.volcano.RelSubset s) {
      n = s.getBestOrOriginal();
    }
    return n;
  }

  static JdbcTableScan firstJdbcScan(RelNode n) {
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

  static String schemaOf(JdbcTableScan scan) {
    List<String> names = scan.getTable().getQualifiedName();
    return names.get(names.size() - 2);
  }
}
