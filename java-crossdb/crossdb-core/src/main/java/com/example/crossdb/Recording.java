package com.example.crossdb;

import javax.sql.DataSource;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.Statement;
import java.util.List;

/** 记录经过的 SQL 与语句配置（setFetchSize/setMaxRows），供自检与单元测试
 * 断言下推行为和参数设置。 */
final class Recording {
  private Recording() {}

  static DataSource dataSource(DataSource delegate, List<String> sqls, List<String> props) {
    return (DataSource) Proxy.newProxyInstance(Recording.class.getClassLoader(),
        new Class<?>[]{DataSource.class},
        (proxy, method, args) -> {
          if (method.getName().equals("getConnection")) {
            Connection c = (Connection) invoke(method, delegate, args);
            return (Connection) Proxy.newProxyInstance(Recording.class.getClassLoader(),
                new Class<?>[]{Connection.class},
                (p2, m2, a2) -> {
                  Object r = invoke(m2, c, a2);
                  if (r instanceof Statement
                      && (m2.getName().startsWith("prepare") || m2.getName().equals("createStatement"))) {
                    if (m2.getName().startsWith("prepare") && a2 != null && a2.length > 0) {
                      sqls.add(String.valueOf(a2[0]));
                    }
                    Class<?> iface = m2.getName().equals("createStatement")
                        ? Statement.class : PreparedStatement.class;
                    return statement((Statement) r, iface, sqls, props);
                  }
                  return r;
                });
          }
          return invoke(method, delegate, args);
        });
  }

  private static Statement statement(Statement target, Class<?> iface,
      List<String> sqls, List<String> props) {
    return (Statement) Proxy.newProxyInstance(Recording.class.getClassLoader(),
        new Class<?>[]{iface},
        (proxy, method, args) -> {
          Object r = invoke(method, target, args);
          String n = method.getName();
          if ((n.equals("executeQuery") || n.equals("execute")) && args != null && args.length > 0) {
            sqls.add(String.valueOf(args[0]));
          }
          if ((n.equals("setFetchSize") || n.equals("setMaxRows") || n.equals("setQueryTimeout"))
              && args != null && args.length > 0) {
            props.add(n + "(" + args[0] + ")");
          }
          return r;
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
