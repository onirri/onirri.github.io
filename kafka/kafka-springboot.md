### SpringBoot整合Kafka

#### 准备工作

1. 提前启动zk，kafka，并且创建一个Topic，具体参照[kafka安装使用](/README.md)
   
2. Maven依赖

```xml
		<dependency>
			<groupId>org.springframework.kafka</groupId>
			<artifactId>spring-kafka</artifactId>
		</dependency>

```
#### 具体实现

```
为了更加体现实际开发需求，一般生产者都是在调用某些接口的服务处理完逻辑之后然后往kafka里面扔数据，然后有一个消费者不停的监控这个Topic，然后处理数据，所以这里把生产者作为一个接口，消费者放到kafka这个目录下，注意@Component注解，不然扫描不到@KafkaListener
```

##### SpringBoot配置文件(application.yml)

```yaml
spring:
  kafka:
    bootstrap-servers: 192.168.0.102:9092
    producer:
      key-serializer: org.apache.kafka.common.serialization.StringSerializer
      value-serializer: org.apache.kafka.common.serialization.StringSerializer
    consumer:
      group-id: test
      enable-auto-commit: true
      auto-commit-interval: 1000
      key-deserializer: org.apache.kafka.common.serialization.StringDeserializer
      value-deserializer: org.apache.kafka.common.serialization.StringDeserializer
```
##### 生成者

```java
package com.sxsoft.testkafka.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 测试kafka生产者
 */
@RestController
@RequestMapping("kafka")
public class TestKafkaProducerController {

    @Autowired
    private KafkaTemplate<String, String> kafkaTemplate;

    @RequestMapping("send")
    public String send(String msg){
        kafkaTemplate.send("nginxlog", msg);
        return "success";
    }

}
```

##### 消费者

```java
package com.sxsoft.testkafka;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

@Component
public class TestConsumer {

    @KafkaListener(topics = "nginxlog")
    public void listen (ConsumerRecord<?, ?> record) throws Exception {
        System.out.printf("topic = %s, offset = %d, value = %s \n", record.topic(), record.offset(), record.value());
    }
}
```

#### 测试

运行项目，执行：http://localhost:8080/kafka/send?msg=hello

控制台输出：

```
topic = nginxlog, offset = 19, value = hello 
```
