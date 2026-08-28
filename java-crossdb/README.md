# crossdb — 基于 Apache Calcite 的跨库 SQL 引擎（MVP）

把多个 JDBC 数据库注册成 schema，用一条 SQL 跨库查询：
- 同库的过滤/投影/JOIN 由 Calcite `JdbcRules` 下推成各库方言 SQL（内置 MySQL/PostgreSQL/Oracle/DB2 等方言翻译）；
- 跨库 JOIN 在本地 Enumerable 执行（hash join）。

自包含子项目，与本仓库其他部分无关，可随时拆成独立仓库。

## 运行自检（两个内存 H2 库，免安装）

```bash
mvn -q compile exec:java -Dexec.mainClass=com.example.crossdb.Main
```

## 接入真实 MySQL / PostgreSQL

```java
try (CrossDb db = new CrossDb()) {
    db.register("shop", buildHikari("jdbc:mysql://host:3306/shop", user, pass));
    db.register("crm",  buildHikari("jdbc:postgresql://host:5432/crm", user, pass));
    ResultSet rs = db.query(
        "SELECT u.name, SUM(o.amount) FROM shop.orders o " +
        "JOIN crm.users u ON u.id = o.user_id GROUP BY u.name");
}
```

驱动已在 pom.xml 里（mysql-connector-j / postgresql），生产建议每个目标库一个独立 HikariCP 连接池。

## 路线图（按需再补，MVP 未实现）

- Bind Join：分批 IN 下推 + 并发拉取（需自定义 Rule + 物理算子）
- 危险 SQL 熔断：拉取行数阈值，超限拒绝执行
- fetchSize 流式拉取，避免整表进堆
- 跨库写事务：Calcite 不提供，需 XA/Seata
