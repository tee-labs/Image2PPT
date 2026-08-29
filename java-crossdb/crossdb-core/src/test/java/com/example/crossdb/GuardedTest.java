package com.example.crossdb;

import org.junit.jupiter.api.Test;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GuardedTest {

  @Test void limitAllowsExactlyThreshold() throws Exception {
    try (Connection c = Fixtures.USERS.getConnection();
        Statement s = c.createStatement();
        ResultSet raw = s.executeQuery("SELECT id FROM users")) {
      ResultSet rs = Guarded.limit(raw, 3);
      int n = 0;
      while (rs.next()) {
        n++;
      }
      assertEquals(3, n);
    }
  }

  @Test void limitThrowsPastThreshold() throws Exception {
    try (Connection c = Fixtures.USERS.getConnection();
        Statement s = c.createStatement();
        ResultSet raw = s.executeQuery("SELECT id FROM users")) {
      ResultSet rs = Guarded.limit(raw, 2);
      assertTrue(rs.next());
      assertTrue(rs.next());
      SQLException e = assertThrows(SQLException.class, rs::next);
      assertTrue(e.getMessage().contains("熔断"), e.getMessage());
    }
  }

  @Test void limitPassesThroughNonNextMethods() throws Exception {
    try (Connection c = Fixtures.USERS.getConnection();
        Statement s = c.createStatement();
        ResultSet raw = s.executeQuery("SELECT id FROM users WHERE id = 1")) {
      ResultSet rs = Guarded.limit(raw, 2);
      assertTrue(rs.next());
      assertEquals(1, rs.getInt(1));
      assertFalse(rs.next());
    }
  }

  @Test void wrapConfiguresStatementsAndGuardsRows() throws Exception {
    DataSource ds = Guarded.wrap(Fixtures.USERS, 7, 2);
    try (Connection c = ds.getConnection(); Statement s = c.createStatement()) {
      assertEquals(7, s.getFetchSize());
      assertEquals(3, s.getMaxRows());
      ResultSet rs = s.executeQuery("SELECT id FROM users");
      assertTrue(rs.next());
      assertTrue(rs.next());
      SQLException e = assertThrows(SQLException.class, rs::next);
      assertTrue(e.getMessage().contains("熔断"), e.getMessage());
    }
    try (Connection c = ds.getConnection();
        PreparedStatement ps = c.prepareStatement("SELECT id FROM users WHERE id = ?")) {
      assertEquals(7, ps.getFetchSize());
      assertEquals(3, ps.getMaxRows());
      ps.setInt(1, 1);
      try (ResultSet rs = ps.executeQuery()) {
        assertTrue(rs.next());
        assertEquals(1, rs.getInt(1));
        assertFalse(rs.next());
      }
    }
  }

  @Test void statementsRegisterForCancelAndUnregisterOnClose() throws Exception {
    Stats stats = new Stats();
    DataSource ds = Guarded.wrap(Fixtures.USERS, 100, 10, "userdb", stats, 0);
    try (Connection c = ds.getConnection();
        PreparedStatement s = c.prepareStatement("SELECT id FROM users WHERE id = ?")) {
      assertEquals(1, stats.live.size(), "语句创建即注册供级联取消");
      s.setInt(1, 1);
      try (ResultSet rs = s.executeQuery()) {
        assertTrue(rs.next());
      }
      stats.cancelAll(); // 对已完成语句 cancel 是无害空操作
      assertEquals(1, stats.live.size());
    }
    assertTrue(stats.live.isEmpty(), "语句关闭后应自动注销");
  }

  @Test void wrapPropagatesTimeoutAndRecordsStats() throws Exception {
    Stats stats = new Stats();
    DataSource ds = Guarded.wrap(Fixtures.USERS, 7, 100, "userdb", stats, 5);
    try (Connection c = ds.getConnection();
        PreparedStatement ps = c.prepareStatement("SELECT id FROM users WHERE id = ?")) {
      assertEquals(5, ps.getQueryTimeout());
      ps.setInt(1, 1);
      try (ResultSet rs = ps.executeQuery()) {
        assertTrue(rs.next());
        assertFalse(rs.next());
      }
    }
    assertEquals(1, stats.schemas.size(), "应记录 userdb 统计");
    Stats.Schema schema = stats.schemas.get("userdb");
    assertEquals(1, schema.rows.sum(), "应记录拉取行数 1");
    assertEquals(1, schema.sqls.size(), "应记录下发 SQL");
    assertTrue(schema.sqls.iterator().next().contains("users"),
        String.valueOf(schema.sqls));
  }
}
