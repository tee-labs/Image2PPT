package com.example.crossdb.spring;

import com.example.crossdb.CrossDb;
import javax.sql.DataSource;

/** 注册数据源到自动装配的 {@link CrossDb}：应用声明一个该类型的 Bean，
 * 在回调里 {@code db.register("shop", shopDataSource)}。 */
@FunctionalInterface
public interface CrossDbCustomizer {
  void customize(CrossDb db) throws Exception;
}
