package com.example.crossdb;

import javax.sql.DataSource;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

/** 危险 SQL 熔断 + fetchSize 流式拉取。
 *
 * <p>注册进 Calcite 的 DataSource 会被包一层代理：每条语句带上
 * fetchSize（流式拉取）与 maxRows=上限+1（驱动侧封顶），
 * 并给所有 ResultSet 加计数代理，拉到的行一旦超过阈值就抛异常拒绝执行。
 */
final class Guarded {
  private Guarded() {}

  static DataSource wrap(DataSource delegate, int fetchSize, long maxRows) {
    return (DataSource) Proxy.newProxyInstance(Guarded.class.getClassLoader(),
        new Class<?>[]{DataSource.class},
        (proxy, method, args) -> {
          if (method.getName().equals("getConnection")) {
            return connectionProxy((Connection) invoke(method, delegate, args), fetchSize, maxRows);
          }
          return invoke(method, delegate, args);
        });
  }

  private static Connection connectionProxy(Connection target, int fetchSize, long maxRows) {
    return (Connection) Proxy.newProxyInstance(Guarded.class.getClassLoader(),
        new Class<?>[]{Connection.class},
        (proxy, method, args) -> {
          Object result = invoke(method, target, args);
          String name = method.getName();
          if (result instanceof Statement
              && (name.equals("createStatement") || name.startsWith("prepareStatement")
                  || name.startsWith("prepareCall"))) {
            Statement st = (Statement) result;
            st.setFetchSize(fetchSize);
            st.setMaxRows((int) Math.min(maxRows + 1, Integer.MAX_VALUE));
            Class<?> iface =
                name.equals("createStatement") ? Statement.class : PreparedStatement.class;
            return statementProxy(st, iface, maxRows);
          }
          return result;
        });
  }

  private static Statement statementProxy(Statement target, Class<?> iface, long maxRows) {
    return (Statement) Proxy.newProxyInstance(Guarded.class.getClassLoader(),
        new Class<?>[]{iface},
        (proxy, method, args) -> {
          Object result = invoke(method, target, args);
          return result instanceof ResultSet ? limit((ResultSet) result, maxRows) : result;
        });
  }

  static ResultSet limit(ResultSet target, long maxRows) {
    long[] seen = {0};
    return (ResultSet) Proxy.newProxyInstance(Guarded.class.getClassLoader(),
        new Class<?>[]{ResultSet.class},
        (proxy, method, args) -> {
          Object result = invoke(method, target, args);
          if (method.getName().equals("next") && Boolean.TRUE.equals(result)
              && ++seen[0] > maxRows) {
            throw new SQLException("crossdb: 拉取行数超过阈值 " + maxRows
                + "，拒绝继续执行（危险 SQL 熔断）");
          }
          return result;
        });
  }

  private static Object invoke(Method method, Object target, Object[] args) throws Exception {
    try {
      return method.invoke(target, args);
    } catch (InvocationTargetException e) {
      Throwable cause = e.getCause();
      if (cause instanceof Exception) {
        throw (Exception) cause;
      }
      throw e;
    }
  }
}
