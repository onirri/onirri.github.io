### Docker Image的安装

#### Docker安装Tomcat

安装Tomcat

```
docker pull tomcat
```
启动Tomcat，并挂载/root/webapps到容器的/usr/local/tomcat/webapps，并设置自动启动
```
docker run -d -p 8080:8080 --name tomcat -v /root/webapps:/usr/local/tomcat/webapps --restart=always tomcat
```

#### Docker安装MySQL

安装MySQL 5.6

```
docker pull mysql:5.6
```
启动MySQL
```
docker run -p 3306:3306 --name mysql -e MYSQL_ROOT_PASSWORD=123456 --restart=always -d mysql:5.6
```
启动MySQL，并挂载conf, logs, data目录 
```
docker run -p 3306:3306 --name mysql -v $PWD/conf:/etc/mysql/conf.d -v $PWD/logs:/logs -v $PWD/data:/var/lib/mysql -e MYSQL_ROOT_PASSWORD=123456 --restart=always -d mysql:5.6
```

#### Docker安装Redis

安装Redis

```
docker pull redis
```
启动Redis
```
docker run -p 6379:6379 --name redis -d redis redis-server --appendonly yes
```
启动Redis，并挂载data目录
```
docker run -p 6379:6379 -v $PWD/data:/data  -d redis:3.2 redis-server --appendonly yes
```

#### Docker安装RabbitMQ

安装RabbitMQ，这里注意获取镜像的时候要获取***management***版本的，不要获取last版本的，***management***版本的才带有管理界面。

```
docker pull rabbitmq:management
```
启动RabbitMQ，默认用户名和密码是guest
```
docker run -d -p 5672:5672 -p 15672:15672 --name rabbitmq rabbitmq:management
```
启动RabbitMQ，设置用户名和密码，挂载data目录
```
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 -v `pwd`/data:/var/lib/rabbitmq --hostname myRabbit -e RABBITMQ_DEFAULT_VHOST=my_vhost  -e RABBITMQ_DEFAULT_USER=admin -e RABBITMQ_DEFAULT_PASS=admin rabbitmq:management
```
