# Transactional Data Change for Real-Time Analytics in Lakehouses

Dongyang Geng Renmin University of China gengdongyang@ruc.edu.cn

Zinuo Li Renmin University of China lizinuo@ruc.edu.cn

## ABSTRACT

Real-time data analytics has become a key demand for lakehouses. It requires propagating data changes from the data source to the lakehouse in real time, while ensuring ACID guarantees and minimizing resource consumption and impact on analytical query performance. This is highly challenging for a data system according to the RUM conjecture, especially given the read-optimized immutable file storage layer in lakehouses. Existing lakehouses ofer only git-style catalog version control on a single table, and they trade data freshness, query performance, and/or resource eficiency to achieve a barely acceptable data-change throughput. In this paper, we propose a novel data synchronization framework, Retina, that enables real-time, cost-eficient, query-performant, and fully transactional data changes in lakehouses. We implement and open-source Retina. Evaluations show that Retina delivers 10-millisecond-level data freshness and over 3.2 million row/s linear-scalable data-change throughput with low resource consumption and query-performance overhead, significantly outperforming state-of-the-art lakehouse systems and lake-compatible HTAP databases.

## PVLDB Reference Format:

Dongyang Geng, Haoqiong Bian, Zinuo Li, and Haoyue Li. Transactional Data Change for Real-Time Analytics in Lakehouses. PVLDB, 19(13): 5148 - 5161, 2026.

doi:10.14778/3849398.3849411

## PVLDB Artifact Availability:

The source code, data, and/or other artifacts have been made available at https://github.com/pixelsdb/pixels/tree/master/pixels-retina.

## 1 INTRODUCTION

Modern data lakes, often called lakehouses [21], bring the advantages of data warehouses in caching, indexing, and metadata management into data lakes, enabling more real-time and performant data analytics on cheap, scalable lake storage composed of open columnar file formats and object storage or Hadoop Distributed File System (HDFS). Real-time analytics requires synchronizing changes to the transactional data source, including tuple inserts,

Haoqiong Bian Renmin University of China bianhaoqiong@ruc.edu.cn

Haoyue Li Renmin University of China lihaoyue03@ruc.edu.cn

deletes, and updates, to the lakehouse with low latency and high throughput. Modern lakehouses such as Apache Iceberg [12] and Paimon [14] employ change-data-capture (CDC) systems such as Debezium [31] to extract data changes from the transaction logs of the data source, and send them to the lakehouse for replay. Such log-based CDC can capture complete, up-to-date data changes with minimal interference to online transactional processing (OLTP), eliminating the expense of extract-transform-load (ETL).

Although CDC solves the extraction issue, the key challenge remains how to promptly replay data changes in the lakehouse with transaction guarantees. Unlike conventional relational databases, the lakehouse prioritizes cost eficiency and scalability, and its storage layer is tuned for analytical queries dominated by large sequential scans and does not support in-place updates [21]. The RUM (Read performance, Update performance, and Memory or storage costs) conjecture proposes that optimizing any two aspects in a data system may negatively impact the third [22]. This implies that, given the lakehouse’s tendency toward low costs and performant reads, simultaneously supporting high-performance updates is highly challenging. For transaction guarantees, a multi-version mechanism is necessary to enable concurrent data changes and queries, as data is not in-place updatable. However, conventional version-chain-based multi-version concurrency control (MVCC) protocols widely used in relational databases [23, 35, 41, 56, 59, 68] are not applicable to lakehouses. Building and traversing version chains in lake storage is ineficient given the immutable file model and high random I/O latency, while keeping chains in memory incurs significant memory overhead and scalability issues.

Existing lakehouse systems, such as Iceberg [12] and Paimon [14], do not support real-time and fully transactional data changes [48]. As a compromise, they implement a git-style catalog version-control mechanism [48]: creating a new snapshot of the table catalog via a checkpoint operation that atomically adds/removes a batch of files to/from the table. This provides limited ACID guarantees on a single table at the file granularity. To replay data changes, these systems use a merge-on-read (MoR) approach, which writes new tuple versions to delta files that are checkpointed in batches. Delta files in the visible catalog snapshot are merged with the base data files to resolve visible tuple versions at query runtime. MoR prioritizes write performance but incurs significant overhead during query execution. Lakehouse systems such as Paimon introduce background compaction to mitigate the merge overhead [15]. However, compaction is CPU- and I/O-intensive. It causes resource contention and back pressure on data-change replay, leading to large fluctuations in replay throughput and data freshness. As shown in Figure 1, we run the data-change workloads extracted from HyBench [72] on Paimon with and without background compaction. We observe that during compaction, data-change throughput drops by approximately half, while checkpoint latency increases by about fourfold, severely reducing data freshness.

![](images/e3b0b86007904bcdbbbb224c79f5c5f20b8dd62c5c3e9f705ee976674c15ca87.jpg)

![](images/f2c389ec94cb5878c27af1811e0d4f911c9d5db3bb502659705ba6ed9c7a7aaf.jpg)  
(a) Data-change throughput.  
(b) Checkpoint latency.  
Figure 1: Performance impact ofcompaction on data changes.

In this paper, we propose Retina, a data synchronization framework for lakehouses that replays the transactional data changes at high throughput and low latency, without compromising query performance or cost eficiency. Based on the observations that storage space is the cheapest resource and that sequential scan is no longer a major performance bottleneck in modern lakehouses, Retina pushes down the visibility check as a vectorized filter operation into the table scan stage of queries. Thus, each scan task performs visibility checks independently on its data split and passes only visible tuple versions downstream. This vectorized-filter-onread (VFoR) approach is far more eficient and scalable than MoR, and it eliminates the expensive compaction of delta and base data. The storage space occupied by stale tuple versions is reclaimed lazily by incremental storage garbage collection jobs.

Retina replays committed write transactions from the data source as ACID mirror transactions in the lakehouse, with the same isolation level and granularity. To enable this, we propose a lightweight MVCC protocol that coordinates transaction replay and query processing using two timestamp watermarks: the high watermark (HWM), which means all mirror transactions earlier than it are replayed and visible to queries, and the low watermark (LWM), which means all analytical queries earlier than it have finished. Each mirror transaction receives a monotonically increasing timestamp and advances HWM as it is replayed. Each query receives (��� − 1) as its snapshot timestamp and advances LWM when it finishes. This protocol isolates mirror transactions and queries into separate timestamp windows, thus avoiding write-read conflicts. Meanwhile, the properties of the CDC stream (see Section 2.2) ensure there are no conflicts among mirror transactions if they obtain timestamps in the order they appear in the stream. Except for timestamp allocation and watermark maintenance, the protocol requires no additional state synchronization or conflict resolution.

The eficiency of VFoR benefits from a novel version-storage scheme that cooperates with the MVCC protocol. We observe that only the tuple versions deleted within (�� �, �� �) might have uncertain visibility across running queries. Therefore, we propose an in-memory data structure, tile visibility, that determines visibility alongside the create timestamp on tuples. It keeps only the delete timestamps of tuple versions deleted after ��� and masks the tuple versions deleted earlier using bitmaps. As �� � advances, old delete timestamps gradually collapse into the bitmaps to free up memory. Checkpoint and recovery mechanisms address durability and long-query stall issues for tile visibility. In evaluations, tile visibility consumes only 1.56 GB (1.16 bit/tuple) of memory for Hy-Bench SF1333 (1TB) under high data-change throughput. Visibility calculation on tile visibility is accelerated by SIMD instructions, adding negligible overhead to query execution.

We implement and open-source Retina as the data synchronization framework in Pixels (https://github.com/pixelsdb/pixels), an eficient lakehouse engine for on-premises and cloud data analytics. In summary, the main contributions of this paper are:

• We reveal that in lakehouses, (1) catalog version control can not provide full ACID guarantees for fine-grained transactions, and (2) MoR and background compaction are the major bottlenecks hindering high-performance, cost-eficient data change and query processing. Hence, we propose a data synchronization framework, Retina, that supports real-time transactional data changes in lakehouses without compromising query performance, scalability, or resource eficiency. (Sections 2 and 3).

• In Retina, we propose a novel version-storage scheme and the corresponding lightweight MVCC protocol to support fine-grained cross-table transactions and eliminate MoR and data compaction. It is based on the observations that (1) filtering out stale tuple versions during table scan is more eficient than MoR+compaction for query processing, and (2) the natural properties of CDC can help simplify MVCC in lakehouses. (Sections 4 and 5).

• Evaluations on HyBench [72] and CH-benCHmark [29] show that Retina simultaneously provides 10-ms-level data freshness, over 3.2 million row/s linear-scalable replay throughput, and transaction guarantees, without compromising query performance or cost-eficiency, significantly outperforms state-of-theart lakehouses (Iceberg [12] and Paimon [14]) and lake-friendly HTAP databases (TiDB [39] and Colibri [56]). (Section 6).

## 2 PRELIMINARIES

In this section, we introduce how data is managed in today’s lakehouses and how data changes are captured by the change-datacapture (CDC) systems and synchronized to lakehouses.

## 2.1 Lakehouse

The lakehouse architecture is proposed primarily to address data governance issues in data lakes [20, 21]. Data lakes have emerged since the early 2010s. They store data in open file formats (e.g., Parquet [17]) on distributed file systems (e.g., HDFS [57]) or cloud object storage (e.g., AWS S3 [4]). Raw data can be quickly loaded into data lakes and analyzed in situ following the schema-on-read paradigm. This enables easy, prompt, cost-eficient, and scalable data access for various applications. However, loose metadata management cannot ensure data consistency during ETL and record updates, leading to data swamps.

Lakehouse table formats, such as DeltaLake [20], Iceberg [12], and Hudi [11], add a table catalog layer to the data lake. Data is still stored in open file formats on lake storage, but the catalog layer provides file-level ACID guarantees for catalog and metadata operations (e.g., adding or removing a batch of files or a column). This is achieved by a logging or snapshot mechanism in the catalog layer [48]. For each catalog operation, the lakehouse performs a checkpoint operation that writes a new log entry or snapshot for the table catalog. Thus, the analytical query can obtain a consistent view of a table’s catalog.

![](images/8f04a0bedab95cefe1adecb464f1868fb14f0e3d8c4ca71ca42d71a516547ddb.jpg)  
(a) Tuple-change message. (b) Transaction message.  
Figure 2: Message structures in log-based CDC (Debezium).

In recent years, lakehouses have been widely adopted for analytical applications across finance, retail, and telecom et.al. In these industries, data freshness is increasingly essential [8, 14, 34], and the data sources are usually transactional systems that undergo random data changes. Real-time data analytics requires synchronizing data changes from data sources to the lakehouse in sub-seconds [34].

## 2.2 Change Data Capture

The first step in real-time data change is to capture fresh datachange events in the data source. Log-based Change data capture (CDC) is designed to address this problem. It captures complete data changes from the write-ahead log or binlog of the source database in real time while introducing minimal impact on OLTP performance. Popular log-based CDC systems include Debezium [31], Kafka Connect [30], Flink CDC [18], and AWS DMS [3] etc.

As an example, Debezium defines a source interface to connect source databases and a sink interface to connect lakehouses. As shown in Figure 2, Debezium captures two message types from the source database. A tuple-change message is generated for the change event (insert, update, or delete) ofeach tuple. In this message, before, after, transaction, and op are the before image (tuple value before change), after image (tuple value after change), transaction information, and operation type (c for insert, u for update, and d for delete), respectively. Before is null for insert, and after is null for delete. A transaction message is generated for each transaction begin or commit (end) event. In this message, id is the transaction ID, and event\_count is the number of tuple changes in the transaction. All messages for a transaction are captured and sent continuously upon transaction commit. The transaction ID in the tuple-change message correlates it to the transaction, and total\_order identifies its order within the transaction. The messages can be passed to the lakehouse through a message queue such as Kafka [43].

## 2.3 Data-Change Replay

Data-change messages arriving at the lakehouse should be replayed on the base data promptly. Unlike most OLTP and HTAP databases, which store data in in-place-updatable formats on high performance storage media such as memory and SSDs, lakehouses store data in open columnar formats on cheap, scalable lake storage. Open columnar formats, such as Parquet [17] and ORC [13], are highly read-optimized. A table stored in these formats is hori zontally split into row groups. Each row group is compressed and stored in columns. One or more row groups are stored in one immutable file in the lake storage. Therefore, to physically update a single row, we must read the file into memory, decompress the corresponding row group, update the column values, recompress the updated row group and file, and rewrite the updated file. This causes significant write amplification and computational overhead.

Moreover, memory eficiency is essential for data-change replay. Analytical queries are memory-intensive because they handle large intermediate results in memory. Meanwhile, lakehouses prioritize cost eficiency, aiming to provide the required performance and capacity at a lower cost. Memory is one of the most expensive resources. It is two orders of magnitude more expensive than lake storage [5, 6]. Therefore, we should minimize the memory footprint of data-change replay to free up memory for analytical queries, thus making it more afordable and applicable in lakehouses.

In addition to cost-eficiency, lakehouses prioritize high scalability and elasticity. Scalability requires no single-point bottleneck in the system. Elasticity requires that the query engine has no heavy states; thus, it can scale with fluctuating query load. Therefore, concurrency control protocols that require maintaining complex data structures (e.g., in-memory version chains [23, 35, 41, 55, 56, 59]) at dedicated servers are not applicable to lakehouses.

Due to these constraints, existing lakehouse systems use a mergeon-read (MoR) approach to replay data changes [11, 12, 14, 20]. MoR writes the new tuple versions to delta files without changing the base files. To commit the delta files atomically, the lakehouse executes a checkpoint operation to create a new snapshot (i.e., version) of the table catalog that includes the delta files. Analytical queries then find the visible snapshot of the table catalog, merge the delta and base files to construct a consistent view of the table. MoR variants include the LSM-style table layout in Apache Paimon [16].

MoR has major limitations that hinder transactional, realtime data-change replay: (1) It has to commit in large batches (e.g., every 5 minutes) because writing delta files and creating a catalog snapshot per mirror transaction is too expensive and would yield unacceptably low replay throughput. (2) It can not provide cross-table ACID guarantees because catalog snapshot is designed to support atomic file operations on a single table. (3) It degrades query performance because reading small delta files and merging them with the base data adds high I/O and CPU cost. Background compaction reduces query costs but is itself expensive and may back-pressure replay, as shown in Figure 1.

HTAP databases with replicated row-column stores (i.e., duallayout), such as TiDB [39] and F1 Lightning [69], also rely on logbased data synchronization. TiDB can deploy its column store, which is similar to Parquet, to lake storage such as S3. It replays logs using DeltaTree, a variant ofMoR with periodic compaction [39]. F1 Lightning also applies MoR with compaction to replay data changes on the column store. Unlike lakehouses, HTAP databases prioritize full transaction guarantees and high freshness over cost eficiency.

## 3 MOTIVATION AND DESIGN

In this section, we first discuss the opportunities for a more eficient solution for transactional, real-time, and low-cost data-change replay in lakehouses. Then, we propose our system architecture and core design considerations.

![](images/b10dbbc88bcab405462cf8e5a6f28fac4ad546edd8a5799c178ec455ff1526c1.jpg)  
Figure 3: Query performance with extra data scan on TPC-H.

## 3.1 Opportunities

In MoR, the ever-growing tuple versions and small delta files are the root cause of the painful trade-of between data freshness and query performance. However, in practice, we observe opportunities to eliminate this trade-of without expensive compaction.

File combination is more eficient for reducing small reads. One main goal of compaction is to reduce small delta-file reads, which are expensive on lake storage [24], for query execution. However, to reduce small reads alone, asynchronously rewriting a group of small files into a regular-sized file is more eficient than compaction, because such a small-file combination does not touch the base data [24, 40]. Our evaluation on AWS S3 shows that 4 EC2 cores can combine small HyBench files of 8192 rows into regular sized files at 1M row/s, high enough in most scenarios. With most delta files combined in time, the query I/O pattern amounts to large sequential scans, which is eficient on lake storage.

Scanning and filtering out invisible tuple versions is efficient. The other goal of compaction is to reclaim stale tuple versions, thus accelerating visibility checks at query runtime. Workloads in lakehouses are typically dominated by table scans [64, 66], which were assumed to be the main bottleneck for query processing on lake storage [24, 38, 66]. However, we find this assumption no longer holds in modern lakehouses. This is because lake storage read bandwidth has improved significantly in recent years [7, 25], and modern columnar file formats are heavily optimized for table scans [2, 9, 10, 24, 26, 44, 71]. In Figure 3, we run TPC-H (SF100) queries on AWS S3 to evaluate the scan and filter performance. The tables are stored in Pixels format, which is optimized for S3 and HDFS [24, 40]. The queries are run by Trino [63] in an m5.8xlarge EC2 instance. We randomly duplicate a percentage of tuples to simulate stale tuple versions, which are then read and filtered out during table scan without passing to downstream stages. The total query latency of all 22 queries increases by only 3.3%, 7.3%, 11.4%, 14.5%, and 19.4% when the stale-tuple ratio is 20%, 40%, 60%, 80%, and 100%, respectively. MoR with periodic compaction in Paimon incurs a 39.0% query slowdown and write stalls in the same environment. These results indicate that scanning a certain percentage of stale tuple versions and filtering them out during table scan is more eficient. This observation creates an opportunity to trade a small amount of storage space, the cheapest resource in lakehouses, for higher data change and query performance.

## 3.2 Architecture Overview

With the observations in Section 3.1, we propose Retina, a realtime transactional data synchronization framework for lakehouses.

![](images/180f8225f36036b351df0b636a5532c44df732414c4b636fc3bfde4bde3e3295.jpg)  
Figure 4: Architectural overview of Retina.

We implement Retina in our open-source lakehouse engine Pixels [53], which has a columnar file format optimized for object storage [24] and HDFS [26, 40]. The Pixels format employs a similar PAX-style [1] file layout as Parquet and ORC, and it does not rely on any special features of Pixels. Thus, migrating Retina to other file formats is practical.

Figure 4 gives an architectural overview of Retina. The datachange messages from CDC are consumed by retina sink (○1 ), which allocates transaction timestamps from the transaction service (○2 ) and reconstructs mirror transactions from the data-change messages (Section 3.3). Inserts, updates, and deletes in the transactions are sharded across a cluster of retina replayers (○3 ). To replay the data-change operations, the replayer invokes the index service to look up and maintain the location of tuples (○4 ). Replayer writes new tuple versions to the write bufer and maintains tuple visibility in the visibility structure. After the replayers successfully replay the data-change operations in a transaction, retina sink requests the transaction service to commit the transaction and advance the high watermark. The replayer flushes the write bufer into the lake storage using the file ID registered from the catalog service (○5 and ○6 ). The write bufer, visibility structure, and data file management are part of the version storage, which is discussed in Section 4.

For an analytical query, the engine obtains the snapshot timestamp from the transaction service (○7 ) and the schema and file list from the catalog service (○8 ) during planning. During table scan, the engine workers request the per-row-group visibility bitmap from the retina replayers (○9 ), which marks whether each tuple version is visible at the query snapshot. Using this bitmap, the reader filters out invisible versions from the row groups read from the columnar files (○10) and returns only visible versions to downstream stages such as joins and aggregations. The write bufer is also read as a special file when it holds tuple versions at or before the query’s snapshot timestamp. We detail query processing in Section 5.

All the components run in their own processes and communicate using remote procedure calls (RPC), except for the index service, which can optionally run in retina replayers for better performance. We provide an interactive animation of Retina in its open-source repository [52] to better explain how Retina works.

## 3.3 Transaction Reconstruction

To support transactional data-change replay, we first reconstruct mirror transactions in retina sink. The messages extracted by the

![](images/9697400fcbdfefd2ffe45600afb9162d0e7a2682730ac0a54d1dfbefeff70934.jpg)  
Figure 5: MVCC algorithm in Retina.

CDC system are naturally ordered by the commit order of source transactions. The messages for a source transaction appear consecutively in the CDC stream. Therefore, we sequentially consume the CDC stream, aggregate messages by transaction ID to form mirror transactions, and assign a unique, monotonically increasing timestamp to each mirror transaction. The timestamp is allocated by the transaction service. We do not reuse the timestamps in CDC messages to avoid timestamp collisions between diferent data sources. Data-change operations in the transactions are sharded by the primary key. Operations in a shard (i.e., bucket) are sent to the corresponding retina replayer in small batches. Retina sink commits the mirror transaction once all its operations are successfully replayed. We apply group begin and commit to reduce RPC costs.

## 3.4 Multi-Version Concurrency Control

Although lakehouses do not execute user transactions, it is necessary to resolve concurrency conflicts among mirror transactions and analytical queries. The concurrency control protocol for lake houses must be cost-eficient, scalable, and comply with the lake storage constraints. This is very challenging.

Retina proposes the watermark-based MVCC in Figure 5. The high watermark (HWM) means all mirror transactions with earlier timestamps are replayed and visible. The low watermark (LWM) means all read-only queries with earlier snapshot timestamps have finished. HWM advances as mirror transactions are committed. An incoming query takes (��� − 1) as its snapshot timestamp and advances LWM as it finishes. A tuple is versioned by the timestamp of the mirror transaction (������\_��) that writes it. This ������\_�� is stored as a hidden column in the columnar files, whereas the ������\_�� of old tuple versions are kept in the visibility structure in retina replayers. Thus, mirror transactions run at or above HWM, queries run in [�� �, �� �), and asynchronous garbage collection runs below LWM to reclaim memory and storage space.

Conflict free. MVCC naturally avoids read-write conflicts (i.e., reads never block later writes). By letting queries run below ���, they avoid seeing inconsistent states of uncommitted mirror transactions, thereby eliminating write-read conflicts.For write-write conflicts, the data source ensures source transactions are committed without conflicts; the CDC system ensures source transactions appear in the CDC stream in their commit order; retina sink ensures timestamps are assigned to their mirror transactions in the same order; and retina replayers ensure that changes to the same shard are relayed in the timestamp order. Mirror transactions therefore can replay in parallel without write-write conflicts.

![](images/3e0a3d43e15f5f217a20d517e89d1112eb1ad3a562109db36354cbf7f634b124.jpg)  
Figure 6: Index scheme in Retina.

Transaction service. The transaction service is the centralized authority for timestamps and watermarks in Retina. The timestamp allocator and the watermarks are atomic integers backed by persistent records in Etcd [62], which is highly available. Thus, the transaction service is fault-tolerant. With engineering optimizations, it supports beginning and committing over 2 million transactions per second on a single node.

## 3.5 Primary Index

To update or delete a tuple, we must locate its primary version in the storage. Retina has a versioned primary index following the same timestamp ordering rule as the version storage; thus, it requires no additional concurrency control. As shown in Figure 6, the primary index consists of a single-point index (SPI) from index keys to logical row IDs and a main index (MI) from logical row IDs to row locations. Similar to Colibri [56], Retina assigns each tuple version a unique, increasing logical row ID. A row location identifies the file, row group, and row ofset of the tuple version. As tuple versions in a file have consecutive row IDs, MI simply maps row ID ranges to row location ranges.

SPI is versioned: a lookup returns the matching entry with the largest timestamp ≤ the request timestamp. A delete is recorded as a tombstone entry. Tombstoned SPI entries and the corresponding row ID ranges in MI are purged asynchronously by the garbage collector in the version storage. The index exposes pluggable interfaces. In this paper, our implementation backs the SPI with RocksDB [33] and the MI with SQLite [36]. We allocate a configurable in-memory block cache to accelerate access to RocksDB. The cache size is 20GB by default and can be tuned according to data-change throughput. The primary index can be partitioned to scale out.

## 4 VERSION STORAGE AND DATA CHANGE

In this section, we introduce the version storage and how data changes and garbage collection are performed on it.

## 4.1 Version Storage

The version storage consists of the column store in the lake storage and the auxiliary data structures in the retina replayers.

Column store. Following the standard storage scheme in lakehouses, a table in Retina is stored as a set of columnar files, each of which contains one or more row groups. The row groups are assigned to table scan tasks to process in parallel. To determine whether a tuple version in a row group is visible during table scan, each tuple version should maintain its create timestamp and delete timestamp (if a later version exists). As the create timestamp does not change after the tuple version is written, we store it as a hidden int64 column in the columnar files, whereas the delete timestamp is maintained in the visibility structure.

![](images/53fc73ad16cee69ccee2580ba1297bf0339b13e2bb9d41d1e630a2a0a9a9a68e.jpg)  
Figure 7: Visibility structure in retina replayer.

Write bufer. Each table has a write bufer in the replayer to accept new tuple versions. The write bufer stores data in row blocks that organize data into column vectors. The write bufer has a single active block that receives new tuple writes and a queue of immutable blocks awaiting background flush to the lake storage. The write bufer is append-only. The active block is atomically turned immutable and replaced by a new one when it is full. Each block has a unique block ID. The blocks flushed to the lake storage are asynchronously combined into normal-sized columnar files as discussed in Section 3.1. The block size (i.e., the number of tuples per block) is configurable. Evaluation shows 4096 is suficient to achieve optimal write performance. With background flush, each write bufer has 1 active and usually 0-2 immutable blocks, consuming less than 2 MB of memory in our experiments.

Visibility structure. To determine tuple visibility, we manage the delete marks and delete timestamps of tuple versions in the visibility structure in replayers. We logically split a row group into tiles, each containing 4096 tuples by default, and create a tile visibility structure for each tile. All tile visibilities for a row group form a RG visibility. As shown in Figure 7, tile visibility has a visibility snapshot and a delete ts array. The former uses a base bitmap to mark whether a tuple in the tile is invisible to queries at or above base ts. The latter is a dynamic array that stores the delete timestamps of tuples deleted after base ts. It is stored as a linked list of delete ts blocks of eight uint64 entries. The tuple ofset in the tile is in the high 12 bits, whereas the tuple’s delete timestamp is in the low 52 bits, which are suficient to allocate timestamps at 1 M/s for 142.8 years. Each block contains 64 bytes, matching a standard CPU cache line and SIMD vector. Tuple deletion after base ts appends an entry to the delete ts array. Queries after base ts can determine tuple visibility by merging delete ts entries into a copy of the base bitmap, accelerated by SIMD. The delete ts array has no random updates; thus, it is lock-free. Memory GC merges its head blocks into the base bitmap and releases memory.

## 4.2 Data-Change Operations

Retina sink sends data-change operations to the replayer through streaming RPC. This ensures that data changes arrive at the replayer in the order of their transaction timestamps.

Insert operation provides the after-image of the new tuple. Retina replayer appends this new tuple to the write bufer and inserts its index entry into the primary index. Blocks in the write bufer form a virtual file with a single row group and corresponding RG visibility. A virtual file has a pre-assigned file ID. When its blocks are combined into a regular file, the file ID is reused. Thus, row locations stay unchanged, and we do not need to update the primary index when write bufer blocks are flushed or combined. We create a new virtual file when the current one is full and combined. As discussed in Section 3.1, such combination is much more eficient than conventional data compaction. The same MVCC protocol applies to both regular and virtual files. Queries can request to read the virtual files, as discussed in detail in Section 5.

Delete operation provides the before image of the tuple. Retina replayer (1) looks up the row location of the tuple’s primary version from the primary index, (2) inserts a tombstone into the primary index, (3) finds the corresponding tile visibility by the row location, and (4) appends the transaction timestamp as the delete timestamp into the delete ts array using atomic memory write. Deletes on the virtual files are no exception. Queries and deletes do not conflict on the delete ts array because queries run below HWM while deletes replay at or above HWM. Delete ts entries later than the snapshot timestamp are ignored by the query.

Update operation provides both before-image and after-image of the tuple. It is a composite operation that deletes the current tuple version and inserts a new one.

## 4.3 Garbage Collection

Retina employs asynchronous garbage collection (GC) to reclaim the memory and storage space used by the stale versions that are no longer visible to any running or future queries.

Memory GC. The growing memory footprint comes from the delete ts in tile visibility. Since all queries run at or above LWM, versions deleted before LWM are globally invisible. The garbage collector periodically checks each tile visibility’s base ts. If it falls behind LWM, the garbage collector merges all delete ts blocks below LWM into a copy of the base bitmap, sets the copy’s base ts to LWM, and swaps the visibility snapshot pointer atomically before freeing the merged blocks. This blocks neither data change nor query.

Visibility checkpoint. Long-running queries may block LWM and memory GC. To avoid this, Retina has a visibility checkpoint mechanism to detect long-running queries, pre-build and materialize the query’s visibility bitmaps to files, force-advance LWM, and resume memory GC. The checkpoint is only 119 MB per billion tuples; reloading it adds negligible overhead for the long-running query. Inactive checkpoints are cleaned asynchronously.

Storage GC. The stale (i.e., globally invisible) tuple versions in the columnar files consume storage space. Since scanning extra data is inexpensive for analytical queries and the storage is cheap in lakehouses, we perform lazy storage GC. During memory GC, if we find that the portion of stale tuple versions in a file exceeds a threshold (e.g., 50%), we push the file ID of the aging file to a pending queue. Retina replayer launches background storage GC jobs on a daily or weekly basis to rewrite aging files and reclaim storage space. A storage GC job chooses a safe snapshot timestamp ��� ≤ ���, below all active visibility checkpoints. Picking two aging files at a time, it rewrites the versions visible at ��� into a new file. To ensure file rewrite does not block transactions and queries, we first write the new file and make it visible only to delete operations. After that, deletes are applied to both new and aging files. During such dual-delete, the GC job constructs the visibility structure and index entries for new files and synchronizes them with those of the aging files. When this is done, we make the new file visible and the aging files invisible to later queries atomically in the catalog service. The GC job will delete the aging files and their visibility structures when no queries are referencing them. Stale index entries for aging files are purged along with the files.

Algorithm 1: Vectorized-Filter-on-Read   
Input: scan split �, query snapshot timestamp ���   
Result: visible tuple versions in �   
1 if � is a regular file split then   
2 foreach row group �� ∈ � do   
3 if ��� < ��.����������� then   
4 continue   
5 � ← GetVisibilityBitmap(��, ���)   
6 if ��� < ��.����������� then   
7 � ← � ∩ FilterCreateTs(��, ���)   
8 ReturnTuples(��, �)   
9 else   
10 (���, ����) ← ReadVirtualFile(�, ���)   
11 � ← GetVisibilityBitmap(�, ���)   
12 foreach flushed block id ��� ∈ ��� do   
13 � ← ReadBlock(���)   
14 ReturnTuples(�, �)   
15 ReturnTuples(����)

## 5 QUERY PROCESSING

This section introduces how Retina participates in query processing.

## 5.1 An Overview

We first briefly explain how Pixels interacts with the query engine. Pixels provides a connector library for the query engine to access the Pixels/Retina services and read the columnar files (Figure 4). Using Trino [63] as an example query engine, during query planning, the Trino coordinator invokes the connector to retrieve the schema and file list of each table from the catalog service. Trino coordinator splits the row groups in the data files into scan splits and assigns them to the Trino workers. During query execution, each Trino worker launches parallel table-scan tasks to scan the assigned splits using the file reader in the connector.

## 5.2 Vectorized-Filter-on-Read

During query planning, the query engine first obtains a snapshot timestamp (i.e., �� � − 1) from the transaction service. This timestamp is passed to table-scan tasks in the scan splits. Vectorizedfilter-on-read (VFoR) then skips invisible tuple versions during table scan and passes each tuple’s visible version downstream. Thus, a query never merges delta and base files or traverses version chains.

As shown in Algorithm 1, lines 2-8 scan a regular columnar file. The file reader first uses the min/max statistics of the hidden ������\_�� column in the row group footers to skip row groups whose min ������\_�� exceeds the query’s snapshot timestamp ��� (lines 3-4). Then it gets the visibility bitmap � from the replayer (line 5). If ��� is within the min and max ������\_��, the reader also filters the row group with the hidden ������\_�� column (lines 6- 7). Finally, it uses � as a vectorized mask to filter tuple versions, returning the visible tuple version to the downstream operators (line 8). Both bitmap-based filtering and timestamp filtering are SIMD-accelerated. If SQL projections and predicates are pushed down to the file reader, they are also applied during VFoR.

As discussed in Section 4.1, Retina wraps the write bufer as a virtual file with a single row group. Lines 10-15 in Algorithm 1 read the virtual file and filter out invisible tuples. The flushed blocks in the virtual file are read from the lake storage (line 13). Only the in-memory blocks are read from the replayer (line 10). The split for the virtual file is scheduled the same as regular splits. It is dispatched to a single worker in the query cluster. No matter how large the query cluster is, the write bufer is read by a single node. Therefore, the write bufer is not a bottleneck for query scalability.

## 5.3 Discussion

Isolation level. Retina ensures that mirror transactions are replayed in the commit order of their source transactions, and the query observes a consistent snapshot of committed mirror transactions. This provides serializable snapshot isolation. However, if we consider the data source and the lakehouse as a monolithic HTAP system, the global isolation level depends on the data source.

CDC delivery. Retina requires that the transaction-grouped datachange messages be delivered to the lakehouse exactly once and in order. This can be guaranteed by modern CDC systems [18, 31]. For CDC setups that do not provide exactly-once and strictlyordered guarantees, retina sink is responsible for de-duplication and ordering using transaction information in the messages.

Fault recovery. The CDC system can persist the recent messages (e.g., in the last 7 days) to a distributed message queue such as Kafka [43]. This is naturally a write-ahead log (WAL) for Retina. When a sink or replayer fails, Retina can redo data changes from the latest WAL checkpoint. The checkpoint is the timestamp and corresponding message ofset at which the index, write bufer, and visibility structures of all mirror transactions before it have been saved to persistent storage. We have implemented this recovery mechanism in Retina. Unlike user data, states including the timestamp/ID allocators, watermarks, and node heartbeats are backed by Etcd [62], a highly available state store.

Point and range query. For point queries, the Retina index service also supports secondary indexes. In non-unique secondary indexes, we store the row ID as the postfix of the SPI index key, and queries search SPI using a prefix without a row ID. Range queries are generally supported by data skipping and range partitioning in lakehouses. Pixels supports data skipping. Range partitioning can be supported by using a range-based sharding policy in retina sink.

## 6 EVALUATION

## 6.1 Experimental Setup

Systems to Evaluate. We evaluate Retina against four state-of-theart systems: Apache Iceberg [12], Apache Paimon [14], TiDB [39], and Colibri [56]. Iceberg is a popular lakehouse engine that employs a MoR approach for data changes. The table format in Iceberg is the de facto standard for today’s lakehouses. We evaluate Iceberg v2. Paimon is a lakehouse engine designed for real-time analytics. It supports fine-grained MoR that partitions tables into buckets and stores each bucket as a columnar LSM-Tree. It has an optimization named the deletion vector (DV) to accelerate MoR. Similar to VFoR, DV can mark tuples in a file as deleted. However, it does not provide snapshot isolation for queries, and it is generated during compaction; thus, it may further degrade data-change performance and data freshness. We evaluate Paimon with DV enabled or disabled. Both Iceberg and Paimon store data in Parquet format. They run as embedded SDKs in compute systems such as Flink [28]. TiDB is a representative HTAP database system with a dual-layout archi tecture. It stores data in row format in its OLTP cluster, TiKV, and in columnar format in its OLAP cluster, TiFlash. Data changes in TiKV are replicated to TiFlash via RAFT [50] logs. This is similar to log-based CDC in lakehouses. TiFlash can store data in a columnar format similar to Parquet in lake storage [39]. Colibri is an academic HTAP database system originating from Umbra [49]. It can store base data in the hybrid Data Block format [46] in object storage.

Compute and Storage. All systems run on AWS EC2 i7i.4xlarge instances, each with 16 vCPUs, 128 GB RAM, 3.75 TB local SSD, and up to 25 Gbps network. Cluster size varies across experiments. All systems store base data in AWS S3 in the same availability zone as the EC2 instances. Local SSD holds the per-system hot state: Retina’s primary index, the state-backend data of Iceberg and Paimon, TiDB’s TiKV row store, and Colibri’s WAL and pages.

Benchmarks. We use HyBench [72] and CH-benCHmark [29], two widely used HTAP benchmarks [51, 56, 58]. To evaluate data-change performance and freshness, we exclude read-only OLTP transac tions, as they do not emit data changes. Then, the OLTP operations in both benchmarks are dominated by random updates (≈ 70%) and inserts (≈ 25%). However, updates in HyBench occur only on small account tables, whereas updates in CH-benCHmark occur primarily on the two largest tables (ORDERLINE and STOCK). We stream data changes to the systems after loading the base datasets. We measure the row throughput in this paper, as it is a common metric for CDC and lakehouses [42, 65]. The average number of rows changed per transaction is 10.23 in HyBench and 13.58 in CH-benchmark.

Baselines. The lakehouse systems (i.e., Retina, Iceberg, and Paimon) ingest data changes from the CDC stream. Thus, they do not pay the costs of OLTP processing. The HTAP systems (i.e., TiDB and Colibri) do not expose a CDC ingestion interface; hence, we have to submit OLTP transactions to their SQL interfaces. This introduces additional SQL parsing and OLTP costs. Therefore, it is not completely fair to directly compare the performance numbers of HTAP and Lakehouse systems. For OLAP queries, Retina, Iceberg, and Paimon use a Trino cluster with one coordinator and five workers unless stated otherwise; TiDB and Colibri use their native engines. We use the recommended settings for all systems. Iceberg uses upserts to reduce write latency. Paimon uses a relaxed compaction trigger to reduce compaction pressure during writes. TiDB creates the indexes required by the workload. We run Iceberg and Paimon in Flink. The catalog checkpoint interval in Flink is varied in the freshness experiment and fixed at 3 min elsewhere. In the single-node experiments, Retina uses one replayer; Iceberg and Paimon each use one Flink node; and Colibri uses one instance. TiDB uses its minimum HTAP deployment of three TiKV nodes and one TiFlash node except in Section 6.4.

## 6.2 Freshness and Throughput

In this experiment, we evaluate the impact of data-change through put on data freshness in a single EC2 instance. We use the HyBench method to measure data freshness: each table carries a ����ℎ\_�� column set at transaction time, a query started at �����\_�� reads max(����ℎ\_��), and freshness is �����\_�� − max(����ℎ\_��).

For Iceberg and Paimon, freshness depends on the catalog checkpoint interval, since replayed changes become visible after checkpointing. We therefore vary it in this experiment. As shown in Figure 8, Retina delivers consistent 10-millisecond-level median data freshness regardless of data-change throughput. On HyBench, the P99 freshness remains below 250 ms. On CH-benCHmark, Retina sustains 80K row/s throughput with a P99 freshness of 103 ms. The freshness is slightly worse at low throughput due to the measurement method. Freshness measurement queries are executed periodically. At low throughput, the time interval between adjacent write transactions is long, leading to an overestimation of freshness.

Apache Iceberg achieves good freshness only at low throughput. On HyBench SF1333, it achieves a P99 freshness of 7.66s at 10K row/s but fails to scale further. The failure stems from I/O latency on object storage. High-frequency commits generate an explosion of small files and catalog snapshots. The I/O bottleneck of writing small files back-pressures the data-change pipeline, causing write stalls. Iceberg performs similarly on CH-benCHmark WH10000, as its simple MoR without compaction is not sensitive to heavy updates on large tables. Apache Paimon sustains high throughput but sacrifices freshness. Paimon (No-DV) exhibits better freshness than Paimon (DV), because DV maintenance introduces significant overhead for the write pipeline. Paimon (DV) fails to run on CH-benCHmark WH10000. Paimon (No-DV) also performs much worse on this workload. This is due to the expense of frequent LSM compaction in the table buckets, which makes Paimon hard to handle heavy updates on large tables. DV maintenance makes compaction performance even worse in this case.

For TiDB, we consulted the TiDB tech team to tune it at best efort. TiDB fails to sustain high throughput. The Raft consensus protocol is the main bottleneck limiting transaction throughput. Compared with lakehouse systems, TiDB excels at providing rowlevel transaction guarantees and better freshness (0.5ms P50 and P99 freshness on HyBench). The data freshness of TiDB is measured by its own performance metric raft write log apply duration. Colibri can only run on HyBench SF100, providing 4K row/s throughput, 545.5ms P50 freshness, and 1092ms P99 freshness, with 16 TP clients. This may be because Colibri is optimized for hotspot updates [56], whereas updates in HyBench are more random.

Summary: Retina is the only system that achieves 10-ms-level freshness under high data-change throughput.

## 6.3 Query Performance Consistency

In this experiment, we evaluate the impact of data changes on query performance. Figure 9 shows the normalized geometric mean of OLAP query latency under a background data-change workload with 1% delta data. Each value is normalized to the corresponding system’s query latency on static base data. The data-change rate is 10K row/s except for Colibri, which caps at 4K row/s.

![](images/33255f3ccc7a71e107c02ec14a85882b899ae3dbc99bb6eae76557d782988ee6.jpg)  
Figure 8: Freshness vs. throughput. Solid lines represent median (P50) freshness, shaded regions represent P99 tail freshness.

![](images/612930d22d15b4f520ff7c5418094de50ce7514a9fa5d17ca2c84ba8045efbe1.jpg)  
Figure 9: Normalized geometric mean of OLAP query latency under 1% delta data and 10K row/s data-change throughput.

Retina demonstrates superior performance consistency across all benchmarks, especially on CH-benCHmark, where heavy updates generate many stale versions. This validates the eficiency of our VFoR design. To further evaluate whether this robustness holds under diferent data-change throughput, we add a separate experiment on HyBench SF1333. We increase the data-change throughput from 10K to 160K row/s, which only Retina and Paimon (No-DV) can sustain on a single node. The normalized query latency for Retina and Paimon (No-DV) remains around 1.04 and 1.33, respectively.

Apache Iceberg exhibits poor query performance on the large datasets. Queries sufer from the many small files and catalog snapshots it produces. Paimon (No-DV) sufers from the I/O amplification of its LSM-Tree structure. Queries must perform expensive merge sorting on numerous small files during execution. Paimon (DV) exhibits consistent performance on HyBench because Hy-Bench updates only small account tables while its large tables receive mostly inserts, so deletion vectors stay small and avoid most merge overhead. On CH-benCHmark WH10000, where updates fall on large tables and generate many versions, Paimon (DV) fails even at 1K row/s. TiDB also fails to execute the queries on CHbenCHmark WH10000. Colibri can only run on HyBench SF100 and exhibits high query performance degradation.

Summary: Retina provides the best query performance consistency under heavy data-change workloads.

## 6.4 Scalability

We measure horizontal scalability by scaling out each system’s datachange components and recording the maximum sustainable datachange throughput on HyBench SF1333. For each cluster size, we run 30-minute trials at increasing throughputs and keep the highest stable rate. Colibri is single-node and excluded in this experiment. For TiDB, we use (TiKV, TiFlash) = (3,1), (6,2), and (12,4) for 4, 8, and 16-node clusters, respectively, as recommended.

![](images/5b7e7c160dfa276c8f38314ffff72b122dfc2c72ce94fb6d0a0fbbe79937bdba.jpg)  
Figure 10: Maximum sustainable data-change throughput under varying cluster size on HyBench SF1333.

As shown in Figure 10, Retina demonstrates linear scalability. With 16 nodes, the throughput reaches 3.2M row/s. This is thanks to the lightweight MVCC mechanism and eficient version storage. Index operations become the main bottleneck during replay. Hence, we scale and co-locate the index service with the replayers. Apache Iceberg does not scale after 4 nodes. This is because committing catalog snapshots becomes the main bottleneck under high concurrency. Paimon (No-DV) also exhibits poor scalability. Although its LSM-Tree design is write-friendly, the massive number of small files triggers S3 request throttling, making it I/O-bounded under high concurrency. Paimon (DV) demonstrates the illusion of linear scalability. This is because maintaining DVs during data compaction incurs substantial memory overhead, and data-change performance is memory-bound on small clusters. The throughput of TiDB plateaus at 11-12K row/s. This is because the heavy updates on the small tables target only a few ranges on a few nodes. The load balancer attempts to ship these hot ranges across the cluster, but this does not help improve throughput.

Furthermore, Retina maintains superior throughput stability as it scales to large clusters. The standard deviation (SD) of its throughput remains below 2% at 16 nodes. The baselines show significant throughput fluctuations (e.g., an SD of 111k at 16 nodes for Paimon No-DV) due to background compaction and write conflicts.

Summary: Retina is the only system that scales linearly to 3.2M row/s stable data-change throughput.

Table 1: Mean resource usage under 10K row/s data-change.
<table><tr><td rowspan="2">System</td><td colspan="2">HyBench SF1333</td><td colspan="2">CH-benCH WH10000</td></tr><tr><td>CPU (%)</td><td>Mem (GB)</td><td>CPU (%)</td><td>Mem (GB)</td></tr><tr><td>Iceberg</td><td>183.05</td><td>39.57</td><td>128.04</td><td>42.19</td></tr><tr><td>Paimon (No-DV)</td><td>817.30</td><td>25.51</td><td>N/A*</td><td>N/A*</td></tr><tr><td>Paimon (DV)</td><td>581.71</td><td>27.67</td><td>N/A*</td><td>N/A*</td></tr><tr><td>TiDB</td><td>2393.79</td><td>84.30</td><td>N/A*</td><td>N/A*</td></tr><tr><td>Retina</td><td>62.17</td><td>19.28</td><td>644.13</td><td>41.34</td></tr></table>

\* N/A indicates the system failed to load or run the benchmark.

## 6.5 Resource Eficiency

In this experiment, we evaluate CPU and memory consumption under a stable data-change throughput on a single node using HyBench and CH-benCHmark. The results are shown in Table 1.

On HyBench SF1333, Retina consumes only 62.17% CPU and 19.28 GB ofmemory. Paimon consumes 581.71% or 817.3% CPU with or without DV, due to frequent background compaction required by the LSM-tree architecture. Iceberg exhibits lower CPU usage at 183.05% but requires 39.57 GB of memory. This stems from the managed memory allocation in its Flink [28] and RocksDB [33] state backends. The resource usage of TiDB includes both TiKV and TiFlash. On CH-benCHmark WH10000, Retina incurs higher CPU usage due to heavy updates on large tables, which trigger frequent seek-and-read operations to retrieve the latest visible index entry from the RocksDB-backed single-point index (SPI). In RocksDB, this is more expensive than point operations on very large datasets. This overhead can be mitigated by partitioning the SPI across more RocksDB column families. Since our indexing framework provides a pluggable interface for index implementations, this issue can be addressed by replacing the index backend with a more eficient one. We also run Colibri with 16 TP clients on HyBench SF100. It consumes 1595.53% CPU and 84.24 GB of memory on average when processing data changes at 4K row/s.

To understand how memory consumption scales with data size, we also perform the experiment on HyBench SF100. From SF100 to SF1333, memory increases from 26.82 GB to 39.57 GB for Iceberg. The larger table increases the states maintained by Flink, leading to a larger runtime memory footprint. Paimon’s memory usage is relatively stable, increasing from 23.20 GB to 25.51 GB without DV and remaining almost unchanged at 27.70 GB and 27.67 GB with DV. This is because Paimon has bounded-size memory tables for the LSM trees. Retina increases from 12.40 GB to 19.28 GB. This increase mainly comes from the growing block cache in the RocksDB index backend, rather than the visibility structure. We can bound this block cache by slightly compromising data-change throughput.

As shown in Figure 11, we further break down the memory consumption of Retina on HyBench SF1333 under 10K row/s datachange throughput. The visibility structure is remarkably lightweight. Its memory footprint is only 1.56 GB, i.e., 1.16 bit/tuple given the 11.6 billion tuples in HyBench SF1333. This footprint remains stable over time, confirming the efectiveness of memory GC. Most memory is used by the index and the Java heap. RocksDB warms up a large block cache to accelerate lookups. The growing Java heap is due to the lazy memory-release mechanism in JVM. S3 bufer is allocated by the AWS SDK. The write bufer has a memory footprint below 2 MB per table per replayer, which is negligible. The memory breakdown on CH-benCHmark WH10000 is similar, with visibility consuming 1.97 GB under the heavy updates.

![](images/04eb941ba44ed0938c42c621fe8084ee0a361289d76899fd5917c1cba5bf6c94.jpg)  
Figure 11: Memory breakdown of Retina on HyBench SF1333.

![](images/3f919934b4592eb4285094d5c3e0e07b96ab0ba97b66f404eb1199e888a7eef4.jpg)

![](images/e349c61a384a522bac97239dcda5ffcb84e2dc6e495ade35747555535583b9c2.jpg)  
Figure 12: Resource usage of Retina under varying datachange throughput on HyBench SF1333.

As shown in Figure 12, as the data-change throughput increases on HyBench SF1333, Retina achieves near-linear CPU usage with an R2 value of 0.98. The marginal cost is only 1.73% CPU per 1K row/s. CH-benCHmark shows higher absolute resource consumption due to complex update patterns. However, Retina still maintains nearlinear scaling, at 3.29% CPU per 1K row/s.

The S3 storage overhead of Retina mainly comes from the unreclaimed stale tuple versions and the hidden timestamp column. With storage GC, stale tuple versions introduce less than 2.4% storage overhead in HyBench evaluations. The hidden column can be compressed efectively as the create timestamps in a file are naturally ordered. For data files bulk-loaded with the same create ts, the storage overhead of the hidden column is 0.18 bit/tuple. For data files created during replay, the storage overhead of the hidden column is 1.85 bit/tuple (1.29% of space) for HyBench and 7.94 bit/- tuple (0.73% of space) for CH-benCHmark. The storage overhead of Iceberg mainly comes from the delta files; hence, it increases with the ratio of delta data. Whereas for Paimon (DV and No-DV), the storage overhead on HyBench can reach 70%-110% mainly due to the storage-space amplification in LSM trees.

Summary: The resource usage of Retina scales linearly with data-change throughput and is lower than that of the baselines in most conditions. The consumption primarily stems from the singlepoint index backend, RocksDB, which is a replaceable component.

## 6.6 Efects of Tuning Knobs

In this experiment, we evaluate the efects of critical tuning knobs in Retina. The evaluations are performed on HyBench SF1333. Data changes are processed on a single node at a data-change throughput of 200K row/s, and queries are executed in a separate Trino cluster with 1 coordinator and 5 workers.

![](images/3057e9f676d3b534a63f077c4b22416bb69b521b0202ff6bd6a4474e0c255014.jpg)  
(a) Data-Change throughput.

![](images/8268e6919ab5ff8a467afeeac17bffb106dc7812f693357e68435cb15e3e092e.jpg)  
(b) Scan query latency.

Figure 13: Efects of write bufer block size.  
![](images/f0e436fb759881ce3424aa447e4447f14c63aa49448d97069e50cd2557048046.jpg)  
Figure 14: Efects of GC interval.

Write bufer block size. Figure 13 shows the efect of write bufer block size on data-change throughput and scan performance. We use a simple table-scan query to evaluate the scan performance. We can see that data-change throughput increases with block size. Larger blocks reduce the frequency of block flushes and amortize the cost of metadata updates. With 4 background block flush threads, data-change throughput saturates at a block size of 4096 rows. At this point, the bottleneck shifts from the write bufer to other replay operations. The replayer must deserialize each incoming data-change request and update the primary index. Such per-tuple costs remain constant regardless of block size. Scan performance remains stable across varying block sizes. As discussed in Section 5.2, the query engine treats the write bufer as virtual files and applies the same VFoR approach to scan them. This improves the scan eficiency and ensures it is not afected by the write bufer block size. We use a default block size of 8192 rows in Retina.

GC interval. Figure 14 shows the efect of memory GC intervals on scan performance. We compare active data change with GC against a static-data baseline. The static baseline has a median scan query latency of 2657 ms. Under active GC, the query latency remains stable at approximately 3100 ms across all intervals from 10 to 240 seconds. The performance overhead remains constant at approximately 450 ms regardless of GC frequency. This proves that the lock-free visibility structure in Retina successfully isolates analytical queries from background maintenance tasks. We use a default memory GC interval of 60 seconds in Retina.

Combination granularity. We evaluate the efect of block combination granularity, i.e., the number of write bufer blocks combined into a regular column file. The results show that it has no obvious efect on data freshness or query performance.

Tile size. Figure 15 shows the efects of visibility tile size. We observe that tile size primarily afects the memory usage of visibility structures. Each tile visibility structure has an inevitable memory overhead for the delete ts array, base ts, and tile visibility ptr. A smaller tile size leads to more tile visibilities and higher memory usage. Such memory usage for visibility approaches the theoretical lower bound of 1 bit per tuple version when the tile size increases. Evaluations also show that query and data-change performance are not afected by tile size. We use 4096 as the default tile size in Retina. It provides favorable memory eficiency and requires 12 bits to store the tuple ofset in the tile. This leaves enough (i.e., 52) bits within the delete ts entry to store the delete timestamp.

![](images/8bc56ea57eda4e19ca7cde0fed92b7d55f83405493e17e584ee3efb1a377bbcd.jpg)  
Figure 15: Efects of visibility tile size.

## 6.7 Ablation Study

In this section, we perform an ablation study to quantify the contributions of the Retina components from diferent dimensions. Since Section 6.5 has analyzed the resource consumption of the components, we focus on transaction guarantees, data-change throughput, data freshness, and query performance in this section.

Transaction guarantees. The catalog snapshot mechanism in existing lakehouses, in fact, implements a git-style serial history for table catalogs rather than a full MVCC for user data [48]. It does not support cross-table transactions, row-level ACID, or concurrent writes [48]. The lightweight MVCC protocol and the corresponding version storage in Retina enable it to replay transactions from OLTP sources and provide full ACID guarantees.

Data-change throughput. Retina can not execute end-to-end data changes if we ablate any essential components. Therefore, we performed unit tests on each component on a single i7i.4xlarge instance to understand its impact on data-change throughput. Results show that: (1) The transaction service can begin and commit 2.67M transactions/s. This is equivalent to 27.31M and 36.26M row/s on HyBench and CH-benCHmark, respectively. (2) Each write bufer can accept 3.37M inserts/s. (3) The visibility structure can process 72.7M deletes/s. (4) A single index service instance can sustain up to 253K updates/s. We can see that the index throughput is close to the 200K row/s max end-to-end data-change throughput on a single node (see Figure 8), while the throughput of the other components is 1-2 orders of magnitude higher. In other words, the index service is not only the main resource consumer (see Section 6.5), but also the main throughput bottleneck. However, indexes are necessary for eficient data updates. The deletion vector in Paimon also requires a primary index to locate the tuple versions. Paimon’s way of storing a table as LSM-Tree buckets provides a form of clustered primary index [48]. However, such a storage layout incurs other bottlenecks as shown in Sections 6.2 and 6.4.

![](images/d000167536a1cdb24a15ac281dc35840d3d1e9b1c2b343ee3ef7bade15b438e2.jpg)  
Figure 16: Data freshness under diferent visibility settings.

Data freshness. The superior data freshness of Retina primarily stems from its novel visibility mechanism. This is more than just allowing queries to read the write bufer; it also requires the appropriate visibility structures and MVCC protocol to ensure data consistency. However, to study its efects on data freshness, as shown in Figure 16, we force the queries not to read the active block in the write bufer (w/o Active Block) or the virtual files (w/o Virtual File) at the risk of violating data consistency. The result confirms that this ability is essential to achieve good data freshness.

Query performance. Section 6.3 demonstrates that Retina has lower query overhead compared to the baselines. In this experiment, we further evaluate the overhead of VFoR compared to the ideal situation of disabling visibility check (i.e., completely not accessing replayers in the queries). The result shows that on HyBench SF100 and a single node, enabling VFoR increases query latency by only 2.33%. This confirms VFoR is close to an optimal visibility solution.

## 7 RELATED WORK

HTAP database systems typically have diferent architectures and application scenarios than real-time lakehouses; however, they share the common goal of improving data freshness for analytical workloads and minimizing performance interference between transaction processing and query execution. In-memory HTAP databases, such as HyPer [41], VEGITO [58], HYRISE [37], and Proteus [54], achieve this goal through dedicated in-memory storage layouts and concurrency-control algorithms. They assume that a majority ofthe dataset fits in memory, making them suitable for applications with small datasets but extremely high performance requirements. Some other HTAP databases, such as SAP HANA [59], Umbra [49], and OceanBase [70], extend their storage engines to high-performance external storage, such as direct-attached SSDs. These types ofHTAP databases difer fundamentally from lakehouses in their underlying storage, leading to significant diferences in architectural design. However, with the wide usage of cloud object storage in OLAP systems [20, 24, 67], some HTAP databases, such as Colibri [56] and TiDB [39], begin to bridge the gap by supporting object storage, which is the common underlying storage in lakehouses, as an optional external storage system. The design considerations in this type of HTAP database ofer valuable insights for lakehouses. For example, the row ID allocation and main index schemes in Colibri are inspiring for the index design in Retina.

Timestamp watermarks are commonly used to implement MVCC in relational databases. For example, the MVCC implementation in HyPer [41], Hekaton [32], SAP HANA [59], and Deuteronomy [47] use a watermark to drive garbage collection (GC). Similar to the low watermark in Retina, transactions in these database systems run above the watermark; thus, stale tuple versions in the version chain that fall behind the watermark can be garbage-collected. This is considered lazy GC in [27], which proposes an approach to eagerly collect stale versions above the watermark. In contrast, we propose the visibility checkpoint approach that can push the low watermark over the snapshot timestamp of long-running queries. This provides favorable GC eficiency while enabling a simpler and more scalable architecture, which is important for lakehouses. Besides, C-Store [61] has an epoch-based watermark that enables snapshot isolation, similar to the epoch-based snapshot isolation in VEGITO [58]. This is inspiring for the MVCC protocol design in Retina. The main contribution of this paper is to propose a lightweight and scalable MVCC mechanism tailored for lakehouses, rather than the watermarks themselves.

Deletion vectors/bitmaps are widely used to accelerate filtering of deleted tuples in column stores and databases such as Paimon [14], Iceberg v3 [19], StarRocks [60], and Colibri [56]. However, deletion bitmaps alone are insuficient for a version storage to determine whether a tuple version is visible to an active query. Therefore, these systems either forgo full transaction support and only support file-level catalog version control (e.g., Paimon, Iceberg v3, and StarRocks) or require additional auxiliary data structures, such as the version chain in Colibri, to support snapshot isolation. In this paper, we propose a novel version storage that supports snapshot isolation while achieving a high memory eficiency close to that of bitmaps. It provides superior performance, scalability, and resource eficiency for lakehouses. In addition to deletion bitmap, Iceberg v2 [19] and Vertica [45] use positional delete file/vector to maintain the locations of deleted tuples, whereas C-Store uses deleted record vector to maintain the epoch in which the tuple was deleted. These approaches are less eficient than deletion bitmap.

## 8 CONCLUSION

This paper proposes a novel data synchronization framework, Retina, for lakehouses. It supports eficient data-change operations without expensive version merging and data compaction, based on the observation that a small amount of additional data scanning and storage usage incurs negligible performance and monetary overheads for modern lakehouses. By making minor, beneficial tradeofs in storage space and sequential reads, we design the lightweight MVCC algorithm and corresponding versioned storage tailored for real-time analytics in lakehouses. Evaluations show that Retina provides 10-ms-level data freshness, over 3.2M row/s scalable replay throughput, and transaction guarantees, without compromising query performance or cost eficiency, significantly outperforming state-of-the-art lakehouses and lake-compatible HTAP databases.

## ACKNOWLEDGMENTS

We thank Professor Anastasia Ailamaki (EPFL) for her constructive suggestions and feedback that significantly improved the quality of this research. We also thank Zhipeng Mao and Tianxiao Shen for their participation in an early-version implementation of Retina during their master’s studies at EPFL. That version provided valu able experience for the final design and implementation of Retina.

## REFERENCES

[1] Anastassia Ailamaki, David J. DeWitt, Mark D. Hill, and Marios Skounakis. 2001. Weaving Relations for Cache Performance. In Proceedings of27th International Conference on Very Large Data Bases (VLDB). Morgan Kaufmann, 169–180.

[2] Alibaba Cloud MaxCompute. 2019. AliORC: A Combination of MaxCompute and Apache ORC. Retrieved August 19, 2026 from https://www.alibabacloud.com/ blog/595359

[3] Amazon Web Services. 2024. Implementing Real-Time CDC Replication: From Relational Databases to Kinesis Data Streams with AWS DMS. Retrieved August 19, 2026 from https://aws.amazon.com/awstv/watch/ea4618992d6/

[4] Amazon Web Services. 2026. Amazon S3. Retrieved August 19, 2026 from https://aws.amazon.com/s3/

[5] Amazon Web Services. 2026. AWS Fargate Pricing. Retrieved August 19, 2026 from https://aws.amazon.com/fargate/pricing/

[6] Amazon Web Services. 2026. AWS S3 Pricing. Retrieved August 19, 2026 from https://aws.amazon.com/s3/pricing/

[7] Amazon Web Services. 2026. How Do I Increase the Maximum Transfer Speed Between Amazon EC2 and Amazon S3? Retrieved August 19, 2026 from https: //repost.aws/knowledge-center/s3-maximum-transfer-speed-ec2

[8] Amazon Web Services. 2026. What is Real-Time Data Streaming? Retrieved August 19, 2026 from https://aws.amazon.com/what-is/real-time-data-streaming/

[9] Apache Software Foundation. 2026. Apache Arrow. Retrieved August 19, 2026 from https://arrow.apache.org

[10] Apache Software Foundation. 2026. Apache CarbonData. Retrieved August 19, 2026 from http://carbondata.apache.org/

[11] Apache Software Foundation. 2026. Apache Hudi. Retrieved August 19, 2026 from https://hudi.apache.org/

[12] Apache Software Foundation. 2026. Apache Iceberg. Retrieved August 19, 2026 from https://iceberg.apache.org/

[13] Apache Software Foundation. 2026. Apache ORC. Retrieved August 19, 2026 from https://orc.apache.org/

[14] Apache Software Foundation. 2026. Apache Paimon. Retrieved August 19, 2026 from https://paimon.apache.org/

[15] Apache Software Foundation. 2026. Apache Paimon - Compaction. Retrieved August 19, 2026 from https://paimon.apache.org/docs/master/primary-key-table/ compaction/

[16] Apache Software Foundation. 2026. Apache Paimon - Overview. Retrieved August 19, 2026 from https://paimon.apache.org/docs/master/primary-key-table

[17] Apache Software Foundation. 2026. Apache Parquet. Retrieved August 19, 2026 from https://parquet.apache.org/

[18] Apache Software Foundation. 2026. Flink CDC. Retrieved August 19, 2026 from https://nightlies.apache.org/flink/flink-cdc-docs-stable

[19] Apache Software Foundation. 2026. Iceberg Table Spec. Retrieved August 19, 2026 from https://iceberg.apache.org/spec

[20] Michael Armbrust, Tathagata Das, Liwen Sun, Burak Yavuz, Shixiong Zhu, Mukul Murthy, Joseph Torres, Herman van Hovell, Adrian Ionescu, Alicja Łuszczak, Michał Świtakowski, Michał Szafrański, Xiao Li, Takuya Ueshin, Mostafa Mokhtar, Peter Boncz, Ali Ghodsi, Sameer Paranjpye, Pieter Senster, Reynold Xin, and Matei Zaharia. 2020. Delta Lake: High-Performance ACID Ta ble Storage over Cloud Object Stores. Proc. VLDB Endow. 13, 12 (2020), 3411–3424. https://doi.org/10.14778/3415478.3415560

[21] Michael Armbrust, Ali Ghodsi, Reynold Xin, and Matei Zaharia. 2021. Lakehouse: A New Generation of Open Platforms that Unify Data Warehousing and Advanced Analytics. In 11th Conference on Innovative Data Systems Research (CIDR).

[22] Manos Athanassoulis, Michael S. Kester, Lukas M. Maas, Radu Stoica, Stratos Idreos, Anastasia Ailamaki, and Mark Callaghan. 2016. Designing Access Meth ods: The RUM Conjecture. In Proceedings ofthe 19th International Conference on Extending Database Technology, EDBT 2016, Bordeaux, France, March 15-16, 2016, Bordeaux, France, March 15-16, 2016. 461–466.

[23] Moshe Battula. 2023. An In-Depth Look at Oracle MySQL HeatWave. Retrieved August 19, 2026 from https://dzone.com/articles/an-in-depth-look-at-oraclemysql-heatwave

[24] Haoqiong Bian and Anastasia Ailamaki. 2022. Pixels: An Eficient Column Store for Cloud Data Lakes. In 38th IEEE International Conference on Data Engineering (ICDE). IEEE, 3078–3090. https://doi.org/10.1109/ICDE53745.2022.00276

[25] Haoqiong Bian, Tiannan Sha, and Anastasia Ailamaki. 2023. Using Cloud Functions as Accelerator for Elastic Data Analytics. Proc. ACM Manag. Data 1, 2 (2023), 161:1–161:27. https://doi.org/10.1145/3589306

[26] Haoqiong Bian, Ying Yan, Wenbo Tao, Liang Jef Chen, Yueguo Chen, Xiaoyong Du, and Thomas Moscibroda. 2017. Wide Table Layout Optimization based on Column Ordering and Duplication. In Proceedings of the 2017 ACM International Conference on Management of Data (SIGMOD). ACM, 299–314. https://doi.org/ 10.1145/3035918.3035930

[27] Jan Böttcher, Viktor Leis, Thomas Neumann, and Alfons Kemper. 2019. Scalable Garbage Collection for In-Memory MVCC Systems. Proc. VLDB Endow. 13, 2 (2019), 128–141. https://doi.org/10.14778/3364324.3364328

[28] Paris Carbone, Asterios Katsifodimos, Stephan Ewen, Volker Markl, Seif Haridi, and Kostas Tzoumas. 2015. Apache Flink™: Stream and Batch Processing in a Single Engine. IEEE Data Eng. Bull. 38, 4 (2015), 28–38.

[29] Richard L. Cole, Florian Funke, Leo Giakoumakis, Wey Guy, Alfons Kemper, Stefan Krompass, Harumi A. Kuno, Raghunath Othayoth Nambiar, Thomas Neumann, Meikel Poess, Kai-Uwe Sattler, Michael Seibold, Eric Simon, and Florian Waas. 2011. The mixed workload CH-benCHmark. In Proceedings of the Fourth International Workshop on Testing Database Systems (DBTest). ACM, Article 8, 6 pages. https://doi.org/10.1145/1988842.1988850

[30] Confluent. 2026. Kafka Connect for Confluent Platform. Retrieved August 19, 2026 from https://docs.confluent.io/platform/current/connect/index.html

[31] Debezium Community. 2026. Debezium. Retrieved August 19, 2026 from https://debezium.io

[32] Cristian Diaconu, Craig Freedman, Erik Ismert, Per-Åke Larson, Pravin Mittal, Ryan Stonecipher, Nitin Verma, and Mike Zwilling. 2013. Hekaton: SQL server’s memory-optimized OLTP engine. In Proceedings of the ACM SIGMOD International Conference on Management of Data. ACM, 1243–1254. https: //doi.org/10.1145/2463676.2463710

[33] Siying Dong, Andrew Kryczka, Yanqin Jin, and Michael Stumm. 2021. RocksDB: Evolution of Development Priorities in a Key-value Store Serving Large-scale Applications. ACM Trans. Storage 17, 4 (2021), 26:1–26:32. https://doi.org/10. 1145/3483840

[34] Dremio. 2026. What is Real-time Data Lake? Retrieved August 19, 2026 from https://www.dremio.com/wiki/real-time-data-lake/

[35] Michael J. Freitag, Alfons Kemper, and Thomas Neumann. 2022. Memory-Optimized Multi-Version Concurrency Control for Disk-Based Database Systems. Proc. VLDB Endow. 15, 11 (2022), 2797–2810. https://doi.org/10.14778/3551793. 3551832

[36] Kevin P. Gafney, Martin Prammer, Larry Brasfield, D. Richard Hipp, Dan Kennedy, and Jignesh M. Patel. 2022. SQLite: Past, Present, and Future. Proc. VLDB Endow. 15, 12 (2022), 3535–3547. https://doi.org/10.14778/3554821.3554842

[37] Martin Grund, Jens Krüger, Hasso Plattner, Alexander Zeier, Philippe Cudre-Mauroux, and Samuel Madden. 2010. HYRISE: a main memory hybrid database storage engine. Proc. VLDB Endow. 4, 2 (2010), 105–116. https://doi.org/10.14778/ 1921071.1921077

[38] Yin Huai, Siyuan Ma, Rubao Lee, Owen O’Malley, and Xiaodong Zhang. 2013. Understanding Insights into the Basic Structure and Essential Issues of Table Placement Methods in Clusters. Proc. VLDB Endow. 6, 14 (2013), 1750–1761. https://doi.org/10.14778/2556549.2556559

[39] Dongxu Huang, Qi Liu, Qiu Cui, Zhuhe Fang, Xiaoyu Ma, Fei Xu, Li Shen, Liu Tang, Yuxing Zhou, Menglong Huang, Wan Wei, Cong Liu, Jian Zhang, Jianjun Li, Xuelian Wu, Lingyu Song, Ruoxi Sun, Shuaipeng Yu, Lei Zhao, Nicholas Cameron, Liquan Pei, and Xin Tang. 2020. TiDB: A Raft-based HTAP Database. Proc. VLDB Endow. 13, 12 (2020), 3072–3084. https://doi.org/10.14778/3415478.3415535

[40] Guodong Jin, Haoqiong Bian, Yueguo Chen, and Xiaoyong Du. 2022. Columnar Storage Optimization and Caching for Data Lakes. In Proceedings ofthe 25th International Conference on Extending Database Technology (EDBT). OpenProceedings.org, 2:419–2:423. https://doi.org/10.48786/EDBT.2022.33

[41] Alfons Kemper and Thomas Neumann. 2011. HyPer: A hybrid OLTP&OLAP main memory database system based on virtual memory snapshots. In Proceedings of the 27th International Conference on Data Engineering (ICDE). IEEE Computer Society, 195–206. https://doi.org/10.1109/ICDE.2011.5767867

[42] Priyansh Khodiyar. 2025. Apache Paimon vs. Apache Iceberg - A Detailed Comparison. Retrieved August 19, 2026 from https://olake.io/iceberg/paimon-vs-iceberg/

[43] Jay Kreps, Neha Narkhede, and Jun Rao. 2011. Kafka: A distributed messaging system for log processing. In Proceedings of the 6th International Workshop on Networking Meets Databases (NetDB).

[44] Maximilian Kuschewski, David Sauerwein, Adnan Alhomssi, and Viktor Leis. 2023. BtrBlocks: Eficient Columnar Compression for Data Lakes. Proc. ACM Manag. Data 1, 2 (2023), 118:1–118:26. https://doi.org/10.1145/3589263

[45] Andrew Lamb, Matt Fuller, Ramakrishna Varadarajan, Nga Tran, Ben Vandiver, Lyric Doshi, and Chuck Bear. 2012. The Vertica Analytic Database: C-Store 7 Years Later. Proc. VLDB Endow. 5, 12 (2012), 1790–1801. https://doi.org/10.14778/ 2367502.2367518

[46] Harald Lang, Tobias Mühlbauer, Florian Funke, Peter Boncz, Thomas Neumann, and Alfons Kemper. 2016. Data Blocks: Hybrid OLTP and OLAP on Compressed Storage using both Vectorization and Compilation. In Proceedings ofthe 2016 International Conference on Management of Data (SIGMOD). ACM, 311–326. https://doi.org/10.1145/2882903.2882925

[47] Justin J. Levandoski, David B. Lomet, Sudipta Sengupta, Ryan Stutsman, and Rui Wang. 2015. High Performance Transactions in Deuteronomy. In Seventh Biennial Conference on Innovative Data Systems Research (CIDR).

[48] Zinuo Li, Dongyang Geng, Haoyue Li, Hailong Yu, Qi Lei, and Haoqiong Bian. 2026. How Consistent and Fresh are Lake Tables Really. In Proceedings ofthe 1st International Workshop on Data FORMATSforModern Architectures and Workloads. ACM, Article 8, 5 pages. https://doi.org/10.1145/3802514.3812602

[49] Thomas Neumann and Michael Freitag. 2020. Umbra: A Disk-Based System with In-Memory Performance. In 10th Conference on Innovative Data Systems Research (CIDR).

[50] Diego Ongaro and John K. Ousterhout. 2014. In Search of an Understandable Consensus Algorithm. In Proceedings of the 2014 USENIX Annual Technical Conference (USENIX ATC). USENIX Association, 305–319.

[51] PingCAP. 2026. How to Run CH-benCHmark Test on TiDB. Retrieved August 19, 2026 from https://docs.pingcap.com/tidb/stable/benchmark-tidb-using-ch/

[52] PixelsDB. 2026. Animation ofExample Operations in Pixels Retina. Retrieved August 19, 2026 from https://github.com/pixelsdb/pixels/tree/master/pixelsretina#visualization

[53] PixelsDB. 2026. Pixels. Retrieved August 19, 2026 from https://github.com/ pixelsdb/pixels

[54] Aunn Raza, Periklis Chrysogelos, Angelos-Christos G. Anadiotis, and Anastasia Ailamaki. 2020. Adaptive HTAP through Elastic Resource Scheduling. In Proceedings ofthe 2020 International Conference on Management ofData (SIGMOD). ACM, 2043–2054. https://doi.org/10.1145/3318464.3389783

[55] Aunn Raza, Periklis Chrysogelos, Angelos-Christos G. Anadiotis, and Anastasia Ailamaki. 2023. One-shot Garbage Collection for In-memory OLTP through Temporality-aware Version Storage. Proc. ACM Manag. Data 1, 1 (2023), 19:1– 19:25. https://doi.org/10.1145/3588699

[56] Tobias Schmidt, Dominik Durner, Viktor Leis, and Thomas Neumann. 2024. Two Birds With One Stone: Designing a Hybrid Cloud Storage Engine for HTAP. Proc. VLDB Endow. 17, 11 (2024), 3290–3303. https://doi.org/10.14778/3681954.3682001

[57] Jefrey Shafer, Scott Rixner, and Alan L. Cox. 2010. The Hadoop distributed filesystem: Balancing portability and performance. In IEEE International Symposium on Performance Analysis ofSystems and Software (ISPASS). IEEE Computer Society, 122–133. https://doi.org/10.1109/ISPASS.2010.5452045

[58] Sijie Shen, Rong Chen, Haibo Chen, and Binyu Zang. 2021. Retrofitting High Availability Mechanism to Tame Hybrid Transaction/Analytical Processing. In 15th USENIXSymposium on Operating Systems Design and Implementation (OSDI). USENIX Association, 219–238.

[59] Vishal Sikka, Franz Färber, Wolfgang Lehner, Sang Kyun Cha, Thomas Peh, and Christof Bornhövd. 2012. Eficient transaction processing in SAP HANA database: the end of a column store myth. In Proceedings of the ACM SIGMOD International Conference on Management of Data. ACM, 731–742. https://doi. org/10.1145/2213836.2213946

[60] StarRocks Project. 2026. StarRocks. Retrieved August 19, 2026 from https: //www.starrocks.io/

[61] Michael Stonebraker, Daniel J. Abadi, Adam Batkin, Xuedong Chen, Mitch Cher niack, Miguel Ferreira, Edmond Lau, Amerson Lin, Samuel Madden, Elizabeth J.

O’Neil, Patrick E. O’Neil, Alex Rasin, Nga Tran, and Stanley B. Zdonik. 2005. C-Store: A Column-oriented DBMS. In Proceedings ofthe 31st International Conference on Very Large Data Bases (VLDB). ACM, 553–564.

[62] The etcd Authors. 2026. etcd. Retrieved August 19, 2026 from https://etcd.io

[63] Trino Software Foundation. 2026. Trino. Retrieved August 19, 2026 from https://trino.io

[64] Alexander van Renen, Dominik Horn, Pascal Pfeil, Kapil Vaidya, Wenjian Dong, Murali Narayanaswamy, Zhengchun Liu, Gaurav Saxena, Andreas Kipf, and Tim Kraska. 2024. Why TPC Is Not Enough: An Analysis of the Amazon Redshif Fleet. Proc. VLDB Endow. 17, 11 (2024), 3694–3706. https://doi.org/10.14778/ 3681954.3682031

[65] Alvar Viana. 2026. Measuring Debezium Server Performance When Streaming MySQL to Kafka. Retrieved August 19, 2026 from https://debezium.io/blog/2026/ 02/02/measuring-debezium-server-performance-mysql-streaming

[66] Lukas Vogel, Alexander van Renen, Satoshi Imamura, Viktor Leis, Thomas Neumann, and Alfons Kemper. 2020. Mosaic: A Budget-Conscious Storage Engine for Relational Database Systems. Proc. VLDB Endow. 13, 11 (2020), 2662–2675. https://doi.org/10.14778/3407790.3407852

[67] Midhul Vuppalapati, Justin Miron, Rachit Agarwal, Dan Truong, Ashish Motivala, and Thierry Cruanes. 2020. Building An Elastic Query Engine on Disaggregated Storage. In 17th USENIX Symposium on Networked Systems Design and Implementation (NSDI). USENIX Association, 449–462.

[68] Yingjun Wu, Joy Arulraj, Jiexi Lin, Ran Xian, and Andrew Pavlo. 2017. An Empirical Evaluation of In-Memory Multi-Version Concurrency Control. Proc. VLDB Endow. 10, 7 (2017), 781–792. https://doi.org/10.14778/3067421.3067427

[69] Jiacheng Yang, Ian Rae, Jun Xu, Jef Shute, Zhan Yuan, Kelvin Lau, Qiang Zeng, Xi Zhao, Jun Ma, Ziyang Chen, Yuan Gao, Qilin Dong, Junxiong Zhou, Jeremy Wood, Goetz Graefe, Jefrey F. Naughton, and John Cieslewicz. 2020. F1 Lightning: HTAP as a Service. Proc. VLDB Endow. 13, 12 (2020), 3313–3325. https://doi.org/ 10.14778/3415478.3415553

[70] Zhifeng Yang, Quanqing Xu, Shanyan Gao, Chuanhui Yang, Guoping Wang, Yuzhong Zhao, Fanyu Kong, Hao Liu, Wanhong Wang, and Jinliang Xiao. 2023. OceanBase Paetica: A Hybrid Shared-nothing/Shared-everything Database for Supporting Single Machine and Distributed Cluster. Proc. VLDB Endow. 16, 12 (2023), 3728–3740. https://doi.org/10.14778/3611540.3611560

[71] Xinyu Zeng, Ruijun Meng, Martin Prammer, Wes McKinney, Jignesh M. Patel, Andrew Pavlo, and Huanchen Zhang. 2025. F3: The Open-Source Data File Format for the Future. Proc. ACM Manag. Data 3, 4 (2025), 245:1–245:27. https: //doi.org/10.1145/3749163

[72] Chao Zhang, Guoliang Li, and Tao Lv. 2024. HyBench: A New Benchmark for HTAP Databases. Proc. VLDB Endow. 17, 5 (2024), 939–951. https://doi.org/10. 14778/3641204.3641206