package com.example.crossdb;

import org.apache.calcite.adapter.enumerable.EnumerableInterpretable;
import org.apache.calcite.adapter.enumerable.EnumerableRel;
import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.jdbc.CalciteConnection;
import org.apache.calcite.jdbc.CalcitePrepare;
import org.apache.calcite.linq4j.Enumerable;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.plan.volcano.RelSubset;
import org.apache.calcite.runtime.Bindable;
import org.apache.calcite.schema.SchemaPlus;

import java.lang.reflect.Proxy;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/** 计划直接执行：用 {@link EnumerableInterpretable#toBindable} 跑优化后的 Enumerable 树，
 * 绕过 RelRunner 的二次 prepare。
 *
 * <p>修复：二次 prepare 会对已优化的树重新做字段裁剪与约定推导，含 WHERE/LIMIT 的
 * 跨库查询会抛 CannotPlanException / EnumerableLimit 断言失败。执行结果以 ResultSet
 * 动态代理暴露（只支持用到的 next/getInt/getString/getObject/close 等方法）。
 */
final class Exec {
  private Exec() {}

  static ResultSet query(CalciteConnection connection, RelNode plan) throws SQLException {
    RelNode rel = unwrap(plan);
    Bindable bindable = EnumerableInterpretable.toBindable(new HashMap<>(), spark(),
        (EnumerableRel) rel, EnumerableRel.Prefer.ARRAY);
    Enumerable<Object[]> rows =
        (Enumerable<Object[]>) bindable.bind(context(connection));
    return resultSet(rows, rel.getRowType().getFieldNames());
  }

  private static RelNode unwrap(RelNode n) {
    while (n instanceof RelSubset s) {
      n = s.getBestOrOriginal();
    }
    return n;
  }

  private static CalcitePrepare.SparkHandler spark() {
    return new CalcitePrepare.SparkHandler() {
      @Override public RelNode flattenTypes(RelOptPlanner planner, RelNode rel, boolean b) {
        return rel;
      }

      @Override public void registerRules(RuleSetBuilder builder) {
      }

      @Override public boolean enabled() {
        return false;
      }

      @Override public org.apache.calcite.runtime.ArrayBindable compile(
          org.apache.calcite.linq4j.tree.ClassDeclaration expr, String s) {
        throw new UnsupportedOperationException();
      }

      @Override public Object sparkContext() {
        return null;
      }
    };
  }

  private static org.apache.calcite.DataContext context(CalciteConnection connection)
      throws SQLException {
    JavaTypeFactory typeFactory = connection.getTypeFactory();
    SchemaPlus rootSchema = connection.getRootSchema();
    return new org.apache.calcite.DataContext() {
      @Override public SchemaPlus getRootSchema() {
        return rootSchema;
      }

      @Override public JavaTypeFactory getTypeFactory() {
        return typeFactory;
      }

      @Override public org.apache.calcite.linq4j.QueryProvider getQueryProvider() {
        return null;
      }

      @Override public Object get(String name) {
        return null;
      }
    };
  }

  private static ResultSet resultSet(Enumerable<Object[]> rows, List<String> names) {
    Iterator<Object[]> it = rows.iterator();
    Object[] current = {null};
    return (ResultSet) Proxy.newProxyInstance(Exec.class.getClassLoader(),
        new Class<?>[]{ResultSet.class},
        (proxy, method, args) -> {
          Object[] cur = current[0] instanceof Object[] ? (Object[]) current[0] : null;
          return switch (method.getName()) {
            case "next" -> {
              if (!it.hasNext()) {
                current[0] = null;
                yield false;
              }
              Object r = it.next();
              // 单列行的 Enumerable 用裸标量表示，统一归一为 Object[]
              current[0] = r instanceof Object[] arr ? arr : new Object[]{r};
              yield true;
            }
            case "close" -> {
              if (it instanceof AutoCloseable closeable) {
                try {
                  closeable.close();
                } catch (Exception ignored) {
                  // 底层迭代器关闭失败不向上传播
                }
              }
              yield null;
            }
            case "wasNull" -> false;
            case "isClosed" -> false;
            case "getInt" -> ((Number) cur[col(args)]).intValue();
            case "getLong" -> ((Number) cur[col(args)]).longValue();
            case "getString" -> {
              Object v = cur[col(args)];
              yield v == null ? null : String.valueOf(v);
            }
            case "getObject" -> cur[col(args)];
            case "findColumn" -> names.indexOf(String.valueOf(args[0]).toUpperCase()) + 1;
            case "getMetaData" -> null;
            default -> throw new SQLException("crossdb: ResultSet 不支持 " + method.getName());
          };
        });
  }

  private static int col(Object[] args) {
    return ((Number) args[0]).intValue() - 1;
  }
}
