package com.example.crossdb.spring;

import com.example.crossdb.CrossDb;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/** 自动装配：提供按 crossdb.* 配置构建的 {@link CrossDb} 单例（生命周期随容器）。
 * 已存在 CrossDb Bean 时不装配（@ConditionalOnMissingBean）；数据源通过
 * {@link CrossDbCustomizer} Bean 注册。 */
@AutoConfiguration
@ConditionalOnMissingBean(CrossDb.class)
@EnableConfigurationProperties(CrossDbProperties.class)
public class CrossDbAutoConfiguration {

  @Bean(destroyMethod = "close")
  public CrossDb crossDb(CrossDbProperties properties,
      org.springframework.beans.factory.ObjectProvider<CrossDbCustomizer> customizer)
      throws Exception {
    CrossDb db = new CrossDb(properties.getFetchSize(), properties.getRowLimit(),
        properties.getBindBatchSize(), properties.getBindParallelism(),
        properties.getQueryTimeout());
    if (properties.isSafeMode()) {
      db.safeMode();
    }
    for (CrossDbCustomizer c : customizer) {
      c.customize(db);
    }
    return db;
  }
}
