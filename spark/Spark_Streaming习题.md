# Spark Streaming习题

1. 什么用于流式数据的处理，使得构建可扩展容错流应用程序变得容易。

 a. Spark SQL

 b. DataFrame

 c. Spark Streaming

 d. GraphX

 

2. 在使用SparkStreaming进行流处理之前需要进行初始化，必须创建一个流上下文对象

a. StreamingContext

b. Discretized Streams

c. Spark Streaming

d. Dstream

 

3. 什么是SparkStreaming提供的最基础的抽象。它表示一系列的数据流，这些数据流可能来自于原始的输入。

 a. StreamingContext

 b. Discretized Streams

 c. Spark Streaming

 d. RDD

 

4. 什么是对DStream中符合条件（符合返回true，否则返回false）的流数据进行筛选并返回DStream类型。

 a. map

 b. flatMap

 c. filter

 d. reduceByKey

 

5. 什么是用来统计DStream源的每个RDD中元素的个数 

a. count

b. flatMap

c. filter

d. reduceByKey



6. 什么是把相同key的DStream聚合在一起。

a. count

b. flatMap

c. filter

d. reduceByKey

 

7. SparkStreaming具有的特点为，为

a. 易于使用

b. 高容错性

c. 高吞吐量

d. 高稳定性



8. SparkStreaming可以接收从什么数据源产生的数据

a. Socket

b. 文件系统

c. Kafka

d. Flume



9. DStreams输出操作包括什么

a. pprint

b. saveAsTextFiles

c. saveAsObjectFiles

d. saveAsHadoopFiles

 

10. DStreams转换操作包括

a. map

b. flatMap

c. filter

d. reduceByKey

 

11. SparkStreaming用于流式数据的处理，使得构建可扩展容错流应用程序变得容易。

a. True

b. False



12. SparkStreaming能和机器学习库（MLlib）以及图计算库（Graphx）进行无缝衔接实现实时在线分析。

a. True

b. False

 

13. Spark Streaming计算过程是将输入的流数据分成多个batch进行处理，从严格意义上来讲spark streaming 并不是一个真正的实时计算框架,因为它是分批次进行处理的。

a. True

b. False



14. Spark Streaming提供了一个高层抽象，称为Discretized Dtream或DStream，它表示连续的数据流。

a. True

b. False

 

15. 在使用SparkStreaming进行流处理之前需要进行初始化，必须创建一个流上下文对象StreamingContext，这是所有SparkStreaming功能的主要入口点。

a. True

b. False

 

16. 在SparkStreaming中可以不容易地在流数据上使用DataFrame和SQL进行操作。

a. True

b. False

 

17. DStream可以通过Kafka，Flume和Kinesis等来源的输入数据流创建，也可以通过在其他DStream上应用高级操作来创建，也可以把DStream看做是一系列RDD。

a. True

b. False

 

18. Spark Streaming是核心Spark API的扩展，它允许实时数据流的可扩展、高通量、容错流处理。

a. True

b. False

 

19. SparkStreaming可以监听某一端口获取数据，通过创建流上下文SparkContext的socketTextStream方法可以直接绑定数据源主机地址和端口。

a. True

b. False

 

20. SparkStreaming除了从套接字端口，监控HDFS外，还可以从kafka、flum等数据源接收并处理数据。

a. True

b. False

 