package com.example.crossdb.spring;

import com.example.crossdb.CrossDb;
import org.springframework.boot.context.properties.ConfigurationProperties;

/** crossdb.* 配置项（默认值与 {@link CrossDb} 内置默认一致）。 */
@ConfigurationProperties("crossdb")
public class CrossDbProperties {
  private int fetchSize = CrossDb.DEFAULT_FETCH_SIZE;
  private long rowLimit = CrossDb.DEFAULT_ROW_LIMIT;
  private int bindBatchSize = CrossDb.DEFAULT_BIND_BATCH_SIZE;
  private int bindParallelism = CrossDb.DEFAULT_BIND_PARALLELISM;
  private int queryTimeout = CrossDb.DEFAULT_QUERY_TIMEOUT;
  private boolean safeMode;

  public int getFetchSize() {
    return fetchSize;
  }

  public void setFetchSize(int fetchSize) {
    this.fetchSize = fetchSize;
  }

  public long getRowLimit() {
    return rowLimit;
  }

  public void setRowLimit(long rowLimit) {
    this.rowLimit = rowLimit;
  }

  public int getBindBatchSize() {
    return bindBatchSize;
  }

  public void setBindBatchSize(int bindBatchSize) {
    this.bindBatchSize = bindBatchSize;
  }

  public int getBindParallelism() {
    return bindParallelism;
  }

  public void setBindParallelism(int bindParallelism) {
    this.bindParallelism = bindParallelism;
  }

  public int getQueryTimeout() {
    return queryTimeout;
  }

  public void setQueryTimeout(int queryTimeout) {
    this.queryTimeout = queryTimeout;
  }

  public boolean isSafeMode() {
    return safeMode;
  }

  public void setSafeMode(boolean safeMode) {
    this.safeMode = safeMode;
  }
}
