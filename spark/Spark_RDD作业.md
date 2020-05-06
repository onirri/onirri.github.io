# 使用RDD的转换和行动命令进行数据分析

## 1.pyspark交互式编程

查看文件“data01.txt”数据集，该数据集包含了某大学计算机系的成绩，数据格式如下所示：

Tom,DataBase,80

Tom,Algorithm,50

Tom,DataStructure,60

Jim,DataBase,90

Jim,Algorithm,60

Jim,DataStructure,80

……

请根据给定的实验数据，在pyspark中通过编程来计算以下内容：

**（1） 该系总共有多少学生；** 

```shell
>>> lines = sc.textFile("file:///usr/local/spark/sparksqldata/data01.txt")
>>> res = lines.map(lambda x:x.split(",")).map(lambda x: x[0]) #获取每行数据的第1列
>>> distinct_res = res.distinct()  #去重操作
>>> distinct_res.count() #取元素总个数
```

 **答案为：265人**

**（2） 该系共开设了多少门课程；** 

```shell
>>> lines = sc.textFile("file:///usr/local/spark/sparksqldata/data01.txt")
>>> res = lines.map(lambda x:x.split(",")).map(lambda x:x[1]) #获取每行数据的第2列
>>> distinct_res = res.distinct() #去重操作
>>> distinct_res.count() #取元素总个数
```

  **答案为8门**

**（3） tdu同学的总成绩平均分是多少；** 

```shell
>>> lines = sc.textFile("file:///usr/local/spark/sparksqldata/data01.txt")
>>> res = lines.map(lambda x:x.split(",")).filter(lambda x:x[0]=="tdu") #筛选Tom同学的成绩信息
>>> res.foreach(print)
>>> score = res.map(lambda x:int(x[2])) #提取Tom同学的每门成绩，并转换为int类型
>>> num = res.count() #Tom同学选课门数
>>> sum_score = score.reduce(lambda x,y:x+y) #Tom同学的总成绩
>>> avg = sum_score/num #总成绩/门数=平均分
>>> print(avg)
```

  **tdu同学的平均分为75.0分**

**（4） 求每名同学的选修的课程门数；** 

```shell
>>> lines = sc.textFile("file:///usr/local/spark/sparksqldata/data01.txt")
>>> res = lines.map(lambda x:x.split(",")).map(lambda x:(x[0],1)) #学生每门课程都对应(学生姓名,1)，学生有n门课程则有n个(学生姓名,1)
>>> each_res = res.reduceByKey(lambda x,y: x+y) #按学生姓名获取每个学生的选课总数
>>> each_res.foreach(print)
```

('wtpb', 4)
('frpq', 4)
('kpgtnv', 8)
('obaety', 4)
('gom', 3)

……

**（5） 该系Spark课程共有多少人选修；** 

```shell
>>> lines = sc.textFile("file:///usr/local/spark/sparksqldata/data01.txt")
>>> res = lines.map(lambda x:x.split(",")).filter(lambda x:x[1]=="Spark")
>>> res.count()
```

  **答案为215人**

**（6） 各门课程的平均分是多少；** 

```shell
>>> lines = sc.textFile("file:///usr/local/spark/sparksqldata/Data01.txt")
>>> res = lines.map(lambda x:x.split(",")).map(lambda x:(x[1],(int(x[2]),1))) #为每门课程的分数后面新增一列1，表示1个学生选择了该课程。格式如('Network', (44, 1))
>>> temp = res.reduceByKey(lambda x,y:(x[0]+y[0],x[1]+y[1])) #按课程名聚合课程总分和选课人数。格式如('Network', (7370, 142))
>>> avg = temp.map(lambda x:(x[0], round(x[1][0]/x[1][1],2))) #课程总分/选课人数 = 平均分，并利用round(x,2)保留两位小数
>>> avg.foreach(print)
```

 ***\*答案为：\****

('Hadoop', 75.89)
('DataBase', 75.08)
('Algorithm', 75.42)
('DataStructure', 75.56)
('Spark', 76.28)
('Python', 76.11)
('Network', 74.64)
('Java', 74.89)

**（7）使用累加器计算共有多少人选了Java这门课。** 

```shell
>>> lines = sc.textFile("file:///usr/local/spark/sparksqldata/data01.txt")
>>> res = lines.map(lambda x:x.split(",")).filter(lambda x:x[1]=="Java") #筛选出选了DataBase课程的数据
>>> accum = sc.accumulator(0) #定义一个从0开始的累加器accum
>>> res.foreach(lambda x:accum.add(1)) #遍历res，每扫描一条数据，累加器加1
>>> accum.value #输出累加器的最终值
```

  **答案：共有35人**

## **2.编写独立应用程序实现数据去重**

对于两个输入文件A和B，编写Spark独立应用程序，对两个文件进行合并，并剔除其中重复的内容，得到一个新文件C。下面是输入文件和输出文件的一个样例，供参考。

输入文件A的样例如下：

20170101   x

20170102   y

20170103   x

20170104   y

20170105   z

20170106   z

输入文件B的样例如下：

20170101   y

20170102   y

20170103   x

20170104   z

20170105   y

根据输入的文件A和B合并得到的输出文件C的样例如下：

20170101   x

20170101   y

20170102   y

20170103   x

20170104   y

20170104   z

20170105   y

20170105   z

20170106   z

  **实验答案参考如下：**

```python
from pyspark import SparkContext
 
#初始化SparkContext
sc = SparkContext('local','remdup')
 
#加载两个文件A和B
lines1 = sc.textFile("file:///usr/local/spark/mycode/remdup/a.txt")
lines2 = sc.textFile("file:///usr/local/spark/mycode/remdup/b.txt")
 
#合并两个文件的内容
lines = lines1.union(lines2)
 
#去重操作
distinct_lines = lines.distinct()
 
#排序操作
res = distinct_lines.sortBy(lambda x:x)
 
#将结果写入result文件中，repartition(1)的作用是让结果合并到一个文件中，不加的话会结果写入到两个文件
res.repartition(1).saveAsTextFile("file:///usr/local/spark/mycode/result/file")
```

 

## **3.编写独立应用程序实现求平均值问题**

每个输入文件表示班级学生某个学科的成绩，每行内容由两个字段组成，第一个是学生名字，第二个是学生的成绩；编写Spark独立应用程序求出所有学生的平均成绩，并输出到一个新文件中。下面是输入文件和输出文件的一个样例，供参考。

Algorithm成绩：

小明 92

小红 87

小新 82

小丽 90

Database成绩：

小明 95

小红 81

小新 89

小丽 85

Python成绩：

小明 82

小红 83

小新 94

小丽 91

平均成绩如下：

(小红,83.67)

(小新,88.33)

(小明,89.67)

(小丽,88.67)

 **实验答案参考如下：**

```python
from pyspark import SparkContext
 
#初始化SparkContext
sc = SparkContext('local',' avgscore')
 
#加载三个文件Algorithm.txt、Database.txt和Python.txt
lines1 = sc.textFile("file:///usr/local/spark/mycode/avgscore/Algorithm.txt")
lines2 = sc.textFile("file:///usr/local/spark/mycode/avgscore/Database.txt")
lines3 = sc.textFile("file:///usr/local/spark/mycode/avgscore/Python.txt")
 
#合并三个文件的内容
lines = lines1.union(lines2).union(lines3)

#为每行数据新增一列1，方便后续统计每个学生选修的课程数目。data的数据格式为('小明', (92, 1))
data = lines.map(lambda x:x.split(" ")).map(lambda x:(x[0],(int(x[1]),1)))
 
#根据key也就是学生姓名合计每门课程的成绩，以及选修的课程数目。res的数据格式为('小明', (269, 3))
res = data.reduceByKey(lambda x,y:(x[0]+y[0],x[1]+y[1]))
 
#利用总成绩除以选修的课程数来计算每个学生的每门课程的平均分，并利用round(x,2)保留两位小数
result = res.map(lambda x:(x[0],round(x[1][0]/x[1][1],2)))
 
#将结果写入result文件中，repartition(1)的作用是让结果合并到一个文件中，不加的话会结果写入到三个文件
result.repartition(1).saveAsTextFile("file:///usr/local/spark/mycode/avgscore/result")
```

## 4.文件说明

数据和参考代码下载[rdd.zip](datas/rdd.zip)

testrdd.py   是pyspark交互式编程源代码

testrdd2.py 是实现数据去重源代码

testrdd3.py 是实现求平均值源代码

createdatas.py 是创建data01.txt数据的源代码

