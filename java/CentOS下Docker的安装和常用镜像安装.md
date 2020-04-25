# CentOS下Docker的安装和常用镜像安装

## 一.Docker的简介

Docker 是一个开源的应用容器引擎，可以让开发者打包他们的应用以及依赖包到一个轻量级、可移植的容器中，然后发布到任何流行的 Linux 机器上，也可以实现虚拟化。容器是完全使用沙箱机制，相互之间不会有任何接口，更重要的是容器性能开销极低。

Docker 从 17.03 版本之后分为 CE（社区版） 和 EE（企业版），我们用社区版就可以了。

![img](images/576507-docker1.png)

| 概念                   | 说明                                                         |
| ---------------------- | ------------------------------------------------------------ |
| Docker 镜像(Images)    | Docker 镜像是用于创建 Docker 容器的模板，比如 Ubuntu 系统。  |
| Docker 容器(Container) | 容器是独立运行的一个或一组应用，是镜像运行时的实体。         |
| Docker 客户端(Client)  | Docker 客户端通过命令行或者其他工具使用 Docker SDK与 Docker 的守护进程通信。 |
| Docker 主机(Host)      | 一个物理或者虚拟的机器用于执行 Docker 守护进程和容器。       |
| Docker 仓库(Registry)  | Docker 仓库用来保存镜像，可以理解为代码控制中的代码仓库。    |
| Docker Machine         | Docker Machine是一个简化Docker安装的命令行工具，通过一个简单的命令行即可在相应的平台上安装Docker |

## 二. CentOS安装Docker

### 2.1 卸载旧版本

较旧的 Docker 版本称为 docker 或 docker-engine 。如果已安装这些程序，请卸载它们以及相关的依赖项。

```sh
$ sudo yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine
```

### 2.2 设置仓库

在新主机上首次安装 Docker Engine-Community 之前，需要设置 Docker 仓库。之后，您可以从仓库安装和更新 Docker。安装所需的软件包。yum-utils 提供了 yum-config-manager ，并且 device mapper 存储驱动程序需要 device-mapper-persistent-data 和 lvm2。

```sh
$ yum install -y yum-utils \
  device-mapper-persistent-data \
  lvm2
```

使用以下命令来设置稳定的仓库

```sh
$ yum-config-manager \
    --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo
```

### 2.3 安装 Docker Engine-Community

安装最新版本的 Docker Engine-Community 和 containerd：

```sh
$ yum install docker-ce docker-ce-cli containerd.io
```

启动 Docker

```sh
$ systemctl start docker
```

通过运行 hello-world 映像来验证是否正确安装了 Docker Engine-Community 。

```sh
$ sudo docker run hello-world
```

### 2.4 Docker镜像加速

编辑 /etc/docker/daemon.json ，写入如下内容：

```json
{"registry-mirrors":["https://registry.docker-cn.com"]}
```

之后重新启动服务：

```sh
$ systemctl daemon-reload
$ systemctl restart docker
```

## 三. Docker Image的安装

### 3.1 Docker安装Tomcat

安装Tomcat

```bash
docker pull tomcat
```

启动Tomcat，并挂载/root/webapps到容器的/usr/local/tomcat/webapps，并设置自动启动

```bash
docker run -d -p 8080:8080 --name tomcat -v /root/webapps:/usr/local/tomcat/webapps --restart=always tomcat
```

### 3.2 Docker安装MySQL

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

### 3.3 Docker安装Redis

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

### 3.4 Docker安装RabbitMQ

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

### 3.5 Docker安装GitLab

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

### 3.6 Docker安装Jenkins

```bash
$ docker pull jenkins/jenkins
$ mkdir /root/jenkins
$ chown -R 1000:1000 /root/jenkins
$ docker run -d -p 8080:8080 -p 50000:50000 -v /root/jenkins:/var/jenkins_home --name jenkins jenkins/jenkins
```

由于国外网站无法访问，Update Site请用https://mirrors.tuna.tsinghua.edu.cn/jenkins/updates/update-center.json，有些插件自动下载安装，请手动下载，http://updates.jenkins-ci.org/download/plugins/

### 3.7 Docker安装memcached

```
docker pull memcached
docker run -d -p 11211:11211 --name memcached memcached
```

