# Spark GraphFrames图计算





## 5. Issue

**1.java.lang.ClassNotFoundException: org.graphframes.GraphFramePythonAPI**

py4j.protocol.Py4JJavaError: An error occurred while calling o48.loadClass.
: java.lang.ClassNotFoundException: org.graphframes.GraphFramePythonAPI
        at java.net.URLClassLoader.findClass(URLClassLoader.java:382)
        at java.lang.ClassLoader.loadClass(ClassLoader.java:418)
        at java.lang.ClassLoader.loadClass(ClassLoader.java:351)
        at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
        at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
        at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
        at java.lang.reflect.Method.invoke(Method.java:498)
        at py4j.reflection.MethodInvoker.invoke(MethodInvoker.java:244)
        at py4j.reflection.ReflectionEngine.invoke(ReflectionEngine.java:357)
        at py4j.Gateway.invoke(Gateway.java:282)
        at py4j.commands.AbstractCommand.invokeMethod(AbstractCommand.java:132)
        at py4j.commands.CallCommand.execute(CallCommand.java:79)
        at py4j.GatewayConnection.run(GatewayConnection.java:238)
        at java.lang.Thread.run(Thread.java:748)

解决方案：

1. 如果使用 pyspark 或 spark-submit 命令，在命令后添加参数--packages，如下代码所示：

  ```shell
$pyspark --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11
  ```

2. 使用 SparkConf 的 spark.jars.packages 属性指定依赖包，如下代码所示：

```python
from pyspark import SparkConf
conf = SparkConf().set('spark.jars.packages', 'graphframes:graphframes:0.6.0-spark2.3-s_2.11')
```


3. 在 SparkSession 中配置，如下代码所示：

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.config('spark.jars.packages', 'graphframes:graphframes:0.6.0-spark2.3-s_2.11').getOrCreate()
```

**2.ERROR ShutdownHookManager: Exception while deleting Spark temp dir**

当我们提交打包好的spark程序时提示如上报错。在windows环境下本身就存在这样的问题，和我们的程序没有关系。若是不想消除该报错，可以在%SPARK_HOME%/conf下的文件log4j.properties添加如下信息：

```properties
log4j.logger.org.apache.spark.util.ShutdownHookManager=OFF
log4j.logger.org.apache.spark.SparkEnv=ERROR
```

