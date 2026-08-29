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
}
