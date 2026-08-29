package com.example.crossdb.spring;

import com.example.crossdb.CrossDb;
import com.example.crossdb.CrossDbUnsafeQueryException;
import org.h2.jdbcx.JdbcDataSource;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.ObjectProvider;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** 不起 Spring 容器，直接验证装配方法与配置绑定逻辑（容器侧由 Spring Boot 标准
 * 机制保证：@AutoConfiguration + imports 文件 + @ConfigurationProperties 绑定）。 */
class CrossDbAutoConfigurationTest {

  @Test void buildsConfiguredCrossDbAndAppliesCustomizer() throws Exception {
    JdbcDataSource h2 = new JdbcDataSource();
    h2.setURL("jdbc:h2:mem:starter;DB_CLOSE_DELAY=-1");
    try (Connection c = h2.getConnection(); Statement s = c.createStatement()) {
      s.execute("CREATE TABLE users(id INT PRIMARY KEY, name VARCHAR(20))");
      s.execute("INSERT INTO users VALUES (1,'alice')");
    }
    CrossDbProperties props = new CrossDbProperties();
    props.setSafeMode(true);
    props.setBindBatchSize(1);
    List<String> registered = new ArrayList<>();
    CrossDb db = new CrossDbAutoConfiguration().crossDb(props,
        providerOf(() -> c -> {
          c.register("userdb", h2);
          registered.add("userdb");
        }));
    try {
      assertEquals(List.of("userdb"), registered, "customizer 应已注册数据源");
      assertThrows(CrossDbUnsafeQueryException.class,
          () -> db.query("SELECT id FROM userdb.users"), "safeMode 配置应生效");
      try (ResultSet rs = db.query("SELECT name FROM userdb.users WHERE id = 1")) {
        assertTrue(rs.next() && "alice".equals(rs.getString(1)));
      }
    } finally {
      db.close();
    }
  }

  /** 最小 ObjectProvider：只支持遍历（装配方法只用了 stream()）。 */
  private static ObjectProvider<CrossDbCustomizer> providerOf(
      java.util.function.Supplier<CrossDbCustomizer> supplier) {
    return new ObjectProvider<>() {
      @Override public Stream<CrossDbCustomizer> stream() {
        return Stream.of(supplier.get());
      }

      @Override public CrossDbCustomizer getObject(Object... args) {
        throw new UnsupportedOperationException();
      }

      @Override public CrossDbCustomizer getObject() {
        throw new UnsupportedOperationException();
      }

      @Override public CrossDbCustomizer getIfAvailable() {
        throw new UnsupportedOperationException();
      }

      @Override public CrossDbCustomizer getIfUnique() {
        throw new UnsupportedOperationException();
      }

      @Override public Iterator<CrossDbCustomizer> iterator() {
        return stream().iterator();
      }
    };
  }
}
