package com.example.crossdb;

import java.sql.SQLException;

/** safeMode 下检测到「无过滤条件的源库全表拉取」时抛出（防呆拦截）。 */
public class CrossDbUnsafeQueryException extends SQLException {
  public CrossDbUnsafeQueryException(String message) {
    super(message);
  }
}
