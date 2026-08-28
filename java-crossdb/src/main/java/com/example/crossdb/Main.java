package com.example.crossdb;

import org.h2.jdbcx.JdbcDataSource;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.LinkedHashMap;
import java.util.Map;

public class Main {
    public static void main(String[] args) throws Exception {
        JdbcDataSource users = ds("jdbc:h2:mem:users;DB_CLOSE_DELAY=-1");
        try (Connection c = users.getConnection(); Statement s = c.createStatement()) {
            s.execute("CREATE TABLE users(id INT PRIMARY KEY, name VARCHAR(50))");
            s.execute("INSERT INTO users VALUES (1,'alice'),(2,'bob'),(3,'carol')");
        }
        JdbcDataSource orders = ds("jdbc:h2:mem:orders;DB_CLOSE_DELAY=-1");
        try (Connection c = orders.getConnection(); Statement s = c.createStatement()) {
            s.execute("CREATE TABLE orders(id INT PRIMARY KEY, user_id INT, amount INT)");
            s.execute("INSERT INTO orders VALUES (100,1,10),(101,2,20),(102,1,5),(103,2,1)");
        }

        try (CrossDb db = new CrossDb()) {
            db.register("userdb", users).register("orderdb", orders);
            try (ResultSet rs = db.query(
                    "SELECT u.name, COUNT(*) AS cnt, SUM(o.amount) AS total " +
                    "FROM userdb.users u JOIN orderdb.orders o ON o.user_id = u.id " +
                    "GROUP BY u.name ORDER BY u.name")) {
                Map<String, String> result = new LinkedHashMap<>();
                while (rs.next()) {
                    String row = rs.getInt(2) + "," + rs.getInt(3);
                    result.put(rs.getString(1), row);
                    System.out.println(rs.getString(1) + "  " + row);
                }
                check("2,15".equals(result.get("alice")));
                check("2,21".equals(result.get("bob")));
                check(!result.containsKey("carol"));
                System.out.println("SELF-CHECK OK: cross-db JOIN + GROUP BY works");
            }
        }
    }

    private static JdbcDataSource ds(String url) {
        JdbcDataSource d = new JdbcDataSource();
        d.setUrl(url);
        return d;
    }

    private static void check(boolean ok) {
        if (!ok) throw new AssertionError("self-check failed");
    }
}
