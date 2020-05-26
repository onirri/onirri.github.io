# CentOS下安装Oracle11g_XE

## 1.下载安装包

https://www.oracle.com/database/technologies/xe-prior-releases.html

## 2.安装步骤

2.1 CentOS是采用的最小安装. 在安装Oracle之前先把依赖装上

```shell
# yum install libaio libaio-devel bc man net-tools -y
```

2.2 分配空间, 如果这里不分配, 后续安装会失败. 就一条一条的执行吧

```shell
# dd if=/dev/zero of=/swapfile bs=1024 count=1048576
# mkswap /swapfile
# swapon /swapfile
# cp /etc/fstab /etc/fstab.backup_$(date +%N)
# echo '/swapfile swap swap defaults 0 0' /etc/fstab
# chown root:root /swapfile
# chmod 0600 /swapfile
# swapon -a
# swapon -s
```

2.3 进入刚才解压好的文件夹 Disk1中, 执行安装

```shell
# cd Disk1/
# rpm -ivh oracle-xe-11.2.0-1.0.x86_64.rpm
```

2.4 安装成功之后, 进行配置

```shell
# /etc/init.d/oracle-xe configure
```

2.5 配置Oracle的环境变量

```shell
# vi /etc/profile
```

```shell
# Oracle Settings
TMP=/tmp; export TMP
TMPDIR=$TMP; export TMPDIR
ORACLE_BASE=/u01/app/oracle; export ORACLE_BASE
ORACLE_HOME=$ORACLE_BASE/product/11.2.0/xe; export ORACLE_HOME
ORACLE_SID=XE; export ORACLE_SID
ORACLE_TERM=xterm; export ORACLE_TERM
PATH=/usr/sbin:$PATH; export PATH
PATH=$ORACLE_HOME/bin:$PATH; export PATH
TNS_ADMIN=$ORACLE_HOME/network/admin
LD_LIBRARY_PATH=$ORACLE_HOME/lib:/lib:/usr/lib; export LD_LIBRARY_PATH
CLASSPATH=$ORACLE_HOME/jlib:$ORACLE_HOME/rdbms/jlib; export CLASSPATH
if [ $USER = "oracle" ]; then
  if [ $SHELL = "/bin/ksh" ]; then
    ulimit -p 16384
    ulimit -n 65536
  else
    ulimit -u 16384 -n 65536
  fi
fi
```

2.6 重新加载配置, 顺便看看配置

```shell
# source /etc/profile
# echo $ORACLE_BASE
```

2.7 安装完成, 用sqlplus上去试试

```shell
# sqlplus system/之前设置的密码@xe
```

