# crossdb — 基于 Apache Calcite 的跨库 SQL 引擎

把多个 JDBC 数据库注册成 schema，用一条 SQL 跨库查询。
定位：**OLTP 点查 / 运维排障** —— 低延迟、低网络 I/O、防慢查与爆内存，纯关系型库、极简开箱即用。

- 同库的过滤/投影/JOIN/聚合由 Calcite `JdbcRules` 下推成各库方言 SQL（内置 MySQL/PostgreSQL/Oracle/DB2 等方言翻译）；
- 跨库 JOIN 走 **Bind Join**：驱动侧 key 分批 `IN` 下推到内表所在库、并发拉取、本地 hash 探测（`BindJoinRule` + `EnumerableBindJoin` + `BindJoinExec`），支持 **复合键（tuple-IN）** 与 **LEFT JOIN**；模式不匹配（复合/非等值残余条件、右表无法整体下推、同库 JOIN 等）时自动退回 Calcite 原生计划；
- **流水线式流式执行**：外表游标按窗口流式读取、每批异步并发拉取，输出流式 yield——驱动侧内存 O(batchSize × parallelism)，不再全量驻内存；
- **Top-N 下推**：`ORDER BY + LIMIT` 且排序列在驱动侧时，一起下推进驱动侧源库 SQL，网络传输降为 O(LIMIT) 量级；
- **传递谓词下推**：`ON a.t = b.t` + `WHERE a.t = 1` 自动把 `b.t = 1` 补到内表侧源库，从源头减少网络传输；
- 所有源库拉取带 **fetchSize 流式读取**、**行数熔断** 与 **queryTimeout 超时传播**（见下）。

自包含子项目，与本仓库其他部分无关，可随时拆成独立仓库。

## 运行自检与单元测试

```bash
mvn test                                                     # 34 个 JUnit 单元测试
mvn -q compile exec:java -Dexec.mainClass=com.example.crossdb.Main   # 端到端自检
```

单元测试覆盖：`Guarded` 熔断与超时/统计（阈值放行 / 超限拒绝 / fetchSize、maxRows、setQueryTimeout、SQL 与行数记录）、`BindJoinExec` 流式执行（多批次并发、去重合批、NULL key、LEFT 补 NULL、复合键 tuple-IN 与 OR 降级、按需拉批、SQL 失败传播、WHERE 构造形态）、`CrossDb` 端到端（JOIN+GROUP BY、WHERE/LIMIT 回归、LEFT JOIN 的 IN 下推、复合键跨库、Top-N 下推、传递谓词下推、safeMode 拦截、超时传播、explain/analyze、行数熔断、非法配置/SQL 拒绝）。自检 Main 覆盖同场景的运行时串联验证。加 `-Dcrossdb.debug=true` 可打印物理计划与规则匹配过程。

## 查询特性

| 特性 | 说明 |
| --- | --- |
| Bind Join | 跨库 INNER/LEFT 等值 JOIN 自动改写：外表 key 每批 `batchSize` 个去重后以 `IN (?)` 下推内表库，`parallelism` 个线程并发拉取，本地 hash 探测；LEFT 时未匹配的外表行右侧补 NULL |
| 复合键 tuple-IN | 多列等值键（`ON a.k1=b.k1 AND a.k2=b.k2`）在 H2/MySQL/PostgreSQL/Oracle 生成 `(k1,k2) IN ((?,?),...)`，其余方言降级为 `(k1=? AND k2=?) OR ...` |
| Top-N 下推 | `ORDER BY <驱动侧列> + LIMIT` 下推进驱动侧源库 SQL（ORDER BY + FETCH/LIMIT），源库只返回 LIMIT 行；本地仍保留 Sort/Limit 保证语义 |
| 传递谓词下推 | 驱动侧 join key 上的常量条件自动补到内表侧源库 SQL |
| 行数熔断 | 每个 `DataSource` 被代理：语句级 `maxRows = 阈值 + 1`（驱动侧封顶），结果集拉取计数超阈值即抛 `SQLException` 拒绝执行；对最终结果和每个源库扫描同样生效 |
| 流式拉取 | 源库语句统一 `setFetchSize(fetchSize)`，逐批读取；Bind Join 外表流式读窗口、输出流式 yield，驱动侧不驻全量 |
| 超时传播 | `queryTimeout` 传播到每条源库语句，慢查询由 JDBC 驱动在源库侧取消，防连接池耗尽 |
| safeMode | `db.safeMode()` 后，计划中出现「无过滤条件的源库全表拉取」直接抛 `CrossDbUnsafeQueryException`（Bind Join 内表例外，其必带 key IN；聚合视为有归约）——OLTP 零容忍全表拉取 |
| explain / analyze | `explain(sql)` 返回优化后物理计划；`analyze(sql)` 执行并输出各源库实际下发的 SQL、每库网络行数、Bind Join 批次与拉取行数 |

配置（构造参数，默认 `1000 / 1_000_000 / 500 / 4 / 0`）：

```java
try (CrossDb db = new CrossDb(fetchSize, rowLimit, bindBatchSize, bindParallelism,
        queryTimeoutSeconds /* 0=不限 */).safeMode() /* 可选 */) { ... }
```

注意：`query(sql)` 返回的 `ResultSet` 只应消费一次（与 JDBC 语义一致），重复迭代会重新下发源库查询；同一 `CrossDb` 同一时刻只支持一条并发查询（CalciteConnection 本身不并发安全），并发统计会串场。

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

驱动已在 pom.xml 里（mysql-connector-j / postgresql），生产建议每个目标库一个独立 HikariCP 连接池（`Guarded` 代理是透明的，直接包住 HikariDataSource 传入即可）。Schema 元数据由 Calcite `JdbcSchema` 在首次用到表时懒加载并缓存于 `CrossDb` 生命周期内，无需额外配置。

## Bind Join 当前边界（触发条件）

- INNER / LEFT JOIN，连接条件**全部**为跨侧等值对（支持多列复合键），无残余非等值条件（有则退回原生计划）；
- 内表可整体下推为一条 JDBC SQL（Scan/Filter/Project 链，且输出列名唯一）；
- 左右两侧来自**不同**已注册库（同库 JOIN 走原生方言下推，不抢）；
- Top-N 要求排序列全部属于 join 的左操作数（驱动侧）；RIGHT/FULL 不做 Bind Join；
- key 经 JDBC `getObject/setObject` 传递；内表哈希表随 distinct key 累积（如需硬上限，先给驱动侧按 key 排序再分批，见下路线图）。

## 路线图（按需再补）

- Statement.cancel 级联取消：当前超时传播覆盖源库保护；外部 cancel 需要跨线程注册表，按需再补
- 内表哈希表每窗口淘汰：仅当驱动侧按 join key 预排序时安全，配合排序下推一起做
- RIGHT / FULL JOIN 的 Bind Join 改写
- 跨库写事务：Calcite 不提供，需 XA/Seata
- Spring Boot / Quarkus 轻量集成模块（当前 `db.register(name, dataSource)` 一行接入已可用）
