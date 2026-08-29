package com.example.crossdb;

import org.h2.jdbcx.JdbcDataSource;

import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;

/** 共享内存 H2 夹具：每个 JVM 初始化一次，数据只读。 */
final class Fixtures {
  static final JdbcDataSource USERS = init("users", """
      CREATE TABLE IF NOT EXISTS users(id INT PRIMARY KEY, name VARCHAR(50));
      CREATE TABLE IF NOT EXISTS small(id INT PRIMARY KEY);
      INSERT INTO users VALUES (1,'alice'),(2,'bob'),(3,'carol');
      INSERT INTO small VALUES (1),(2);
      """);
  static final JdbcDataSource ORDERS = init("orders", """
      CREATE TABLE IF NOT EXISTS orders(id INT PRIMARY KEY, user_id INT, amount INT);
      INSERT INTO orders VALUES (100,1,10),(101,2,20),(102,1,5),(103,2,1);
      """);
  /** 复合键夹具：(user_id, tenant_id) 两列联合 join。 */
  static final JdbcDataSource CREDS = init("creds", """
      CREATE TABLE IF NOT EXISTS creds(user_id INT, tenant_id INT, login VARCHAR(20));
      INSERT INTO creds VALUES (1,100,'a1'),(2,100,'b1'),(1,200,'a2'),(3,100,'c1');
      """);
  static final JdbcDataSource QUOTAS = init("quotas", """
      CREATE TABLE IF NOT EXISTS quotas(tenant_id INT, user_id INT, quota INT);
      INSERT INTO quotas VALUES (100,1,10),(100,2,20),(200,1,5);
      """);

  private Fixtures() {}

  private static JdbcDataSource init(String name, String ddl) {
    JdbcDataSource d = new JdbcDataSource();
    d.setURL("jdbc:h2:mem:crossdb_" + name + ";DB_CLOSE_DELAY=-1");
    try (Connection c = d.getConnection(); Statement s = c.createStatement()) {
      for (String part : ddl.split(";")) {
        if (!part.isBlank()) {
          s.execute(part);
        }
      }
    } catch (SQLException e) {
      throw new IllegalStateException("夹具初始化失败: " + name, e);
    }
    return d;
  }
}
