package com.example.crossdb;

import org.apache.calcite.adapter.jdbc.JdbcSchema;
import org.apache.calcite.jdbc.CalciteConnection;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Properties;

public class CrossDb implements AutoCloseable {
    private final CalciteConnection connection;

    public CrossDb() throws SQLException {
        Properties info = new Properties();
        info.setProperty("lex", "MYSQL");
        Connection raw = DriverManager.getConnection("jdbc:calcite:", info);
        this.connection = raw.unwrap(CalciteConnection.class);
    }

    public CrossDb register(String schema, DataSource dataSource) throws SQLException {
        connection.getRootSchema().add(schema,
                JdbcSchema.create(connection.getRootSchema(), schema, dataSource, null, null));
        return this;
    }

    public ResultSet query(String sql) throws SQLException {
        return connection.createStatement().executeQuery(sql);
    }

    @Override
    public void close() throws SQLException {
        connection.close();
    }
}
