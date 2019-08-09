### Ngnix负载均衡

http节点下添加：

 upstream backend {

   server 127.0.0.1:8080 weight=5;

   server 127.0.0.1:8081;

 }

location节点下添加：

​        location / {

​            root html;

​            index index.html index.htm;

​            proxy_pass http://backend;

​        }

负载均衡的5种算法

https://www.cnblogs.com/DarrenChan/p/8967412.html