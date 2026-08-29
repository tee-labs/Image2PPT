# crossdb — 基于 Apache Calcite 的跨库 SQL 引擎

把多个 JDBC 数据库注册成 schema，用一条 SQL 跨库查询。
定位：**OLTP 点查 / 运维排障** —— 低延迟、低网络 I/O、防慢查与爆内存，纯关系型库、极简开箱即用。

- 同库的过滤/投影/JOIN/聚合由 Calcite `JdbcRules` 下推成各库方言 SQL（内置 MySQL/PostgreSQL/Oracle/DB2 等方言翻译）；
- 跨库 JOIN 走 **Bind Join**：驱动侧 key 分批 `IN` 下推到内表所在库、并发拉取、本地 hash 探测（`BindJoinRule` + `EnumerableBindJoin` + `BindJoinExec`），支持 **INNER/LEFT/RIGHT/FULL** 与 **复合键（tuple-IN）**；RIGHT 交换内外侧按 LEFT 形态执行（行型恢复原始列序），FULL 在外表耗尽后对内表补一次分块 `NOT IN` 反连接；模式不匹配（复合/非等值残余条件、右表无法整体下推、同库 JOIN 等）时自动退回 Calcite 原生计划；
- **流水线式流式执行**：外表游标按窗口流式读取、每批异步并发拉取，输出流式 yield——驱动侧内存 O(batchSize × parallelism)，不再全量驻内存；外表按 join key 有序时（排序下推场景）自动**每窗口淘汰**哈希旧条目；
- **Top-N 下推**：`ORDER BY + LIMIT` 且排序列在驱动侧时，一起下推进驱动侧源库 SQL，网络传输降为 O(LIMIT) 量级；
- **传递谓词下推**：`ON a.t = b.t` + `WHERE a.t = 1` 自动把 `b.t = 1` 补到内表侧源库，从源头减少网络传输；
- 所有源库拉取带 **fetchSize 流式读取**、**行数熔断** 与 **queryTimeout 超时传播**（见下）。

自包含子项目（Maven 多模块：`crossdb-core` + `crossdb-spring-boot-starter`），与本仓库其他部分无关，可随时拆成独立仓库。

## 运行自检与单元测试

```bash
mvn test                                                     # 43 个 JUnit 单元测试（两个模块）
mvn -q -pl crossdb-core exec:java -Dexec.mainClass=com.example.crossdb.Main   # 端到端自检
```

单元测试覆盖：`Guarded` 熔断与超时/统计（阈值放行 / 超限拒绝 / fetchSize、maxRows、setQueryTimeout、SQL 与行数记录、在途语句取消注册表）、`BindJoinExec` 流式执行（多批次并发、去重合批、NULL key、LEFT/RIGHT 行序、FULL 反连接、复合键 tuple-IN 与 OR 降级、按需拉批、排序淘汰、SQL 失败传播、WHERE 构造形态）、`CrossDb` 端到端（JOIN+GROUP BY、WHERE/LIMIT 回归、LEFT/RIGHT/FULL 的 IN 下推、复合键跨库、Top-N 下推、传递谓词下推、safeMode 拦截、超时传播、级联取消、explain/analyze、行数熔断、非法配置/SQL 拒绝）。自检 Main 覆盖同场景的运行时串联验证。加 `-Dcrossdb.debug=true` 可打印物理计划与规则匹配过程。

## 查询特性

| 特性 | 说明 |
| --- | --- |
| Bind Join | 跨库 INNER/LEFT/RIGHT/FULL 等值 JOIN 自动改写：外表 key 每批 `batchSize` 个去重后以 `IN (?)` 下推内表库，`parallelism` 个线程并发拉取，本地 hash 探测；LEFT/FULL 时未匹配的外表行补 NULL，FULL 再对内表补一次分块 `NOT IN` 反连接；RIGHT 交换内外侧执行、行型保持原始 [左 ++ 右] |
| 复合键 tuple-IN | 多列等值键（`ON a.k1=b.k1 AND a.k2=b.k2`）在 H2/MySQL/PostgreSQL/Oracle 生成 `(k1,k2) IN ((?,?),...)`，其余方言降级为 `(k1=? AND k2=?) OR ...` |
| Top-N 下推 | `ORDER BY <驱动侧列> + LIMIT` 下推进驱动侧源库 SQL（ORDER BY + FETCH/LIMIT），源库只返回 LIMIT 行；本地仍保留 Sort/Limit 保证语义 |
| 传递谓词下推 | 驱动侧 join key 上的常量条件自动补到内表侧源库 SQL |
| 哈希窗口淘汰 | 驱动侧按 join key 有序（如排序下推后的计划）时，每窗口合并前淘汰更小的 key，内表哈希内存从 O(distinct keys) 降为 O(窗口 keys)；检测保守，宁可不淘汰不错杀匹配 |
| 行数熔断 | 每个 `DataSource` 被代理：语句级 `maxRows = 阈值 + 1`（驱动侧封顶），结果集拉取计数超阈值即抛 `SQLException` 拒绝执行；对最终结果、每个源库扫描和 FULL 反连接同样生效 |
| 流式拉取 | 源库语句统一 `setFetchSize(fetchSize)`，逐批读取；Bind Join 外表流式读窗口、输出流式 yield，驱动侧不驻全量 |
| 超时传播 | `queryTimeout` 传播到每条源库语句，慢查询由 JDBC 驱动在源库侧取消，防连接池耗尽 |
| 级联取消 | `db.cancel()`（外部线程调用）遍历在途源库语句逐个 `Statement.cancel()`，主动掐断慢查询；语句关闭自动注销注册表 |
| safeMode | `db.safeMode()` 后，计划中出现「无过滤条件的源库全表拉取」直接抛 `CrossDbUnsafeQueryException`（Bind Join 内表例外，其必带 key IN 过滤；聚合视为有归约）——OLTP 零容忍全表拉取 |
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

- INNER/LEFT/RIGHT/FULL JOIN，连接条件**全部**为跨侧等值对（支持多列复合键），无残余非等值条件（有则退回原生计划）；RIGHT 交换内外侧执行，FULL 的反连接会把外表 distinct key 全量攒内存（受行数熔断封顶）；
- 内表可整体下推为一条 JDBC SQL（Scan/Filter/Project 链，且输出列名唯一）；
- 左右两侧来自**不同**已注册库（同库 JOIN 走原生方言下推，不抢）；
- Top-N 要求排序列全部属于 join 的左操作数（驱动侧），仅 INNER/LEFT；
- key 经 JDBC `getObject/setObject` 传递；哈希淘汰仅在驱动侧按 key 有序时启用。

## Spring Boot 接入

`crossdb-spring-boot-starter` 模块提供自动装配（Spring Boot 3.x）：

```properties
# application.properties（均有默认值，可不配）
crossdb.fetch-size=1000
crossdb.row-limit=1000000
crossdb.bind-batch-size=500
crossdb.bind-parallelism=4
crossdb.query-timeout=0
crossdb.safe-mode=true
```

```java
// 数据源通过 Customizer Bean 注册，注入已有的 HikariCP DataSource 即可
@Bean
CrossDbCustomizer crossDbCustomizer(DataSource shopDs) {
  return db -> db.register("shop", shopDs);
}
```

已声明 `CrossDb` Bean 时不装配；关闭时自动 `db.close()`（destroyMethod）。

## 路线图（按需再补）

- ~~Statement.cancel 级联取消~~ ✅ `db.cancel()`
- ~~RIGHT / FULL JOIN 的 Bind Join 改写~~ ✅ RIGHT 交换执行 + FULL NOT IN 反连接
- ~~内表哈希表每窗口淘汰~~ ✅ 排序驱动侧自动启用
- ~~Spring Boot 轻量集成~~ ✅ `crossdb-spring-boot-starter`；Quarkus 需要时再加
- 跨库写事务：Calcite 不提供，需引入 XA/Seata 级别的外部组件，超出本项目「纯查询、零侵入」定位，明确不做；有真实诉求时建议在应用层用 Saga/补偿，而不是下沉到查询引擎
