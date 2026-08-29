# crossdb — 基于 Apache Calcite 的跨库 SQL 引擎

把多个 JDBC 数据库注册成 schema，用一条 SQL 跨库查询：
- 同库的过滤/投影/JOIN/聚合由 Calcite `JdbcRules` 下推成各库方言 SQL（内置 MySQL/PostgreSQL/Oracle/DB2 等方言翻译）；
- 跨库 JOIN 走 **Bind Join**：驱动侧 key 分批 `IN` 下推到内表所在库、并发拉取后本地 hash 探测（`BindJoinRule` + `EnumerableBindJoin`），内表不再被全量拉取；模式不匹配（非 INNER、复合/非等值条件、右表无法整体下推、同库 JOIN 等）时自动退回 Calcite 原生计划；
- 所有源库拉取带 **fetchSize 流式读取** 与 **行数熔断**（见下）。

自包含子项目，与本仓库其他部分无关，可随时拆成独立仓库。

## 运行自检与单元测试

```bash
mvn test                                                     # 20 个 JUnit 单元测试
mvn -q compile exec:java -Dexec.mainClass=com.example.crossdb.Main   # 端到端自检
```

单元测试覆盖：`Guarded` 熔断（恰好等于阈值放行 / 超限拒绝 / fetchSize 与 maxRows 语句配置）、`BindJoinExec` 分批（batch=1 多批次 / 去重合批 / NULL key 跳过 / 空外表不下发查询 / SQL 失败传播）、`CrossDb` 端到端（跨库 JOIN+GROUP BY、Bind Join 的 IN 下推替换内表全量扫描、复合连接条件与 LEFT JOIN 回退原生计划、同库 JOIN 下推、行数熔断、fetchSize 传播、非法配置/SQL 拒绝）。自检 Main 覆盖同场景的运行时串联验证。加 `-Dcrossdb.debug=true` 可打印物理计划与规则匹配过程。

## 查询特性

| 特性 | 说明 |
| --- | --- |
| Bind Join | 跨库 INNER 等值 JOIN 自动改写：外表 key 每批 `batchSize` 个去重后以 `IN (?)` 下推内表库，`parallelism` 个线程并发拉取，本地 hash 探测 |
| 行数熔断 | 每个 `DataSource` 被代理：语句级 `maxRows = 阈值 + 1`（驱动侧封顶），结果集拉取计数超阈值即抛 `SQLException` 拒绝执行；对最终结果和每个源库扫描同样生效 |
| 流式拉取 | 源库语句统一 `setFetchSize(fetchSize)`，逐批读取，避免整表进堆 |

配置（构造参数，默认 `1000 / 1_000_000 / 500 / 4`）：

```java
try (CrossDb db = new CrossDb(fetchSize, rowLimit, bindBatchSize, bindParallelism)) { ... }
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

驱动已在 pom.xml 里（mysql-connector-j / postgresql），生产建议每个目标库一个独立 HikariCP 连接池（`Guarded` 代理是透明的，直接包住 HikariDataSource 传入即可）。

## Bind Join 当前边界（触发条件）

- INNER JOIN，连接条件为**单个**跨侧等值对（`a.x = b.y`），两侧 key 的 SQL 类型一致；
- 内表（等号内侧任一方向皆可，优化器按代价挑驱动侧）可整体下推为一条 JDBC SQL（Scan/Filter/Project，且输出列名唯一）；
- 左右两侧来自**不同**已注册库（同库 JOIN 走原生方言下推，不抢）；
- 驱动侧全量驻内存——大外表需调大 `bindBatchSize` 或先过滤；key 经 JDBC `getObject/setObject` 传递。

## 路线图（按需再补）

- 跨库写事务：Calcite 不提供，需 XA/Seata
