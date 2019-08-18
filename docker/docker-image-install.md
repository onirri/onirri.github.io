### Docker Image的安装

#### Docker安装Tomcat

安装Tomcat

```bash
docker pull tomcat
```
启动Tomcat，并挂载/root/webapps到容器的/usr/local/tomcat/webapps，并设置自动启动
```bash
docker run -d -p 8080:8080 --name tomcat -v /root/webapps:/usr/local/tomcat/webapps --restart=always tomcat
```

#### Docker安装MySQL

安装MySQL 5.6

```bash
docker pull mysql:5.6
```
启动MySQL
```bash
docker run -p 3306:3306 --name mysql -e MYSQL_ROOT_PASSWORD=123456 --restart=always -d mysql:5.6
```
启动MySQL，并挂载conf, logs, data目录 
```bash
docker run -p 3306:3306 --name mysql -v $PWD/conf:/etc/mysql/conf.d -v $PWD/logs:/logs -v $PWD/data:/var/lib/mysql -e MYSQL_ROOT_PASSWORD=123456 --restart=always -d mysql:5.6
```

#### Docker安装Redis

安装Redis

```bash
docker pull redis
```
启动Redis
```bash
docker run -p 6379:6379 --name redis -d redis redis-server --appendonly yes
```
启动Redis，并挂载data目录
```bash
docker run -p 6379:6379 -v $PWD/data:/data  -d redis:3.2 redis-server --appendonly yes
```

#### Docker安装RabbitMQ

安装RabbitMQ，这里注意获取镜像的时候要获取***management***版本的，不要获取last版本的，***management***版本的才带有管理界面。

```bash
docker pull rabbitmq:management
```
启动RabbitMQ，默认用户名和密码是guest
```bash
docker run -d -p 5672:5672 -p 15672:15672 --name rabbitmq rabbitmq:management
```
启动RabbitMQ，设置用户名和密码，挂载data目录
```bash
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 -v `pwd`/data:/var/lib/rabbitmq --hostname myRabbit -e RABBITMQ_DEFAULT_VHOST=my_vhost  -e RABBITMQ_DEFAULT_USER=admin -e RABBITMQ_DEFAULT_PASS=admin rabbitmq:management
```

后台管理地址：http://ip:8080/

#### Docker安装GitLab

```bash
$ docker pull gitlab/gitlab-ce
$ sudo docker run --detach \
  --hostname 192.168.64.129 \
  --publish 443:443 --publish 80:80 --publish 222:22 \
  --name gitlab \
  --restart always \
  --volume /root/gitlab/config:/etc/gitlab \
  --volume /root/gitlab/logs:/var/log/gitlab \
  --volume /root/gitlab/data:/var/opt/gitlab \
  gitlab/gitlab-ce
```

在config目录下添加gitlab.rb

```properties
# 配置http协议所使用的访问地址,不加端口号默认为80
external_url 'http://192.168.64.130'

# 配置ssh协议所使用的访问地址和端口
gitlab_rails['gitlab_ssh_host'] = '192.168.64.130'
gitlab_rails['gitlab_shell_ssh_port'] = 222 # 此端口是run时22端口映射的222端口
```

生成SSH Key，打开gitlab,找到Profile Settings-->SSH Keys--->Add SSH Key,并把上一步中复制的内容粘贴到Key所对应的文本框

```
$ ssh-keygen -t rsa -C "email@example.com"
```

### Docker安装Jenkins

```bash
$ docker pull jenkins/jenkins
$ mkdir /root/jenkins
$ chown -R 1000:1000 /root/jenkins
$ docker run -d -p 8080:8080 -p 50000:50000 -v /root/jenkins:/var/jenkins_home --name jenkins jenkins/jenkins
```

由于国外网站无法访问，Update Site请用https://mirrors.tuna.tsinghua.edu.cn/jenkins/updates/update-center.json，有些插件自动下载安装，请手动下载，http://updates.jenkins-ci.org/download/plugins/

### Docker安装memcached

```
docker pull memcached
docker run -d -p 11211:11211 --name memcached memcached
```

