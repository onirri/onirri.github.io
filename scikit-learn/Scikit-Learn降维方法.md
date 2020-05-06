#                                             Scikit-Learn降维方法

## 1. 降维方法

降维是指减少原数据的维度，我们回忆一下我们比较熟悉的鸢尾花数据，在这个数据集中，数据包含了四个维度，四个维度确实不多，但是如果现实中我们拿到的数据有成千上万的维度呢？这些维度里所有的数据都应该计算吗？如果都计算那么我们可以想象其消耗的时间是非常巨大的，因此在此我们引入了降维的方法，常用的降维方法有：

（1）主成分分析（PCA） :

​        在 PCA 中，数据从原来的坐标系转换到新的坐标系，新坐标系的选择是由数据本身决定的。第一个新坐标轴选择的是原始数据中方差最大的方向，第二个新坐标轴选择和第一个坐标轴正交且具有最大方差的方向。该过程一直重复，重复次数为原始数据中特征的数目。我们会发现，大部分方差都包含在最前面的几个新坐标轴中。因此，我们可以忽略余下的坐标轴，即对数据进行降维处理。

![datasetline.png](images/20170813103204269.png)

**注：降维后的参数是经过旋转得到的新特征空间，已经不是以前的那种具体参数。**

（2）因子分析（Factor Analysis）

​      在因子分析中，我们假设在观察数据的生成中有一些观察不到的隐变量（latent variable）。假设观察数据是这些隐变量和某些噪声数据的线性组合。那么隐变量的数据可能比观察数据的数目少，也就是说通过找到隐变量就可以实现数据降维。

​      例如： 考察一个人的整体情况，就直接组合3样成绩（观测数据，存在：数学、语文、英语成绩），看平均成绩（隐变量）就行。

（3）独立成分分析（ICA） 

​      ICA方法认为数据是从 N 个数据源混合组成的，这一点和因子分析有些类似。ICA的做法就是寻找出这些独立数据源，这些数据源之间在统计上是相互独立的，如果独立数据源数目少于观察数据的数目，就可以实现降维。

## 2.PCA

### 2.1. scikit-learn PCA类介绍

​        在scikit-learn中，与PCA相关的类都在sklearn.decomposition包中。最常用的PCA类就是sklearn.decomposition.PCA，我们下面主要也会讲解基于这个类的使用的方法。

​        除了PCA类以外，最常用的PCA相关类还有KernelPCA类，它主要用于非线性数据的降维，需要用到核技巧。因此在使用的时候需要选择合适的核函数并对核函数的参数进行调参。

　　另外一个常用的PCA相关类是IncrementalPCA类，它主要是为了解决单机内存限制的。有时候我们的样本量可能是上百万+，维度可能也是上千，直接去拟合数据可能会让内存爆掉， 此时我们可以用IncrementalPCA类来解决这个问题。IncrementalPCA先将数据分成多个batch，然后对每个batch依次递增调用partial_fit函数，这样一步步的得到最终的样本最优降维。　　

### 2.2. sklearn.decomposition.PCA参数介绍

　　下面我们主要基于sklearn.decomposition.PCA来讲解如何使用scikit-learn进行PCA降维。PCA类基本不需要调参，一般来说，我们只需要指定我们需要降维到的维度，或者我们希望降维后的主成分的方差和占原始维度所有特征方差和的比例阈值就可以了。

　　现在我们对sklearn.decomposition.PCA的主要参数做一个介绍：

　　1）**n_components**：这个参数可以帮我们指定希望PCA降维后的特征维度数目。

​        最常用的做法是直接指定降维到的维度数目，此时n_components是一个大于等于1的整数。

​        当然，我们也可以指定主成分的方差和所占的最小比例阈值，让PCA类自己去根据样本特征方差来决定降维到的维度数，此时n_components是一个（0，1]之间的数。

​         当然，我们还可以将参数设置为"mle", 此时PCA类会用MLE算法根据特征的方差分布情况自己去选择一定数量的主成分特征来降维。

​         我们也可以用默认值，即不输入n_components，此时n_components=min(样本数，特征数)。

　　2）**whiten** ：判断是否进行白化。

​        所谓白化，就是对降维后的数据的每个特征进行归一化，让方差都为1.对于PCA降维本身来说，一般不需要白化。如果你PCA降维后有后续的数据处理动作，可以考虑白化。默认值是False，即不进行白化。

　　3）**svd_solver**：即指定奇异值分解SVD的方法，由于特征分解是奇异值分解SVD的一个特例，一般的PCA库都是基于SVD实现的。

​        有4个可以选择的值：{‘auto’, ‘full’, ‘arpack’, ‘randomized’}。

​       randomized一般适用于数据量大，数据维度多同时主成分数目比例又较低的PCA降维，它使用了一些加快SVD的随机算法。 

​        full则是传统意义上的SVD，使用了scipy库对应的实现。

​      arpack和randomized的适用场景类似，区别是randomized使用的是scikit-learn自己的SVD实现，而arpack直接使用了scipy库的sparse SVD实现。

​       默认是auto，即PCA类会自己去在前面讲到的三种算法里面去权衡，选择一个合适的SVD算法来降维。一般来说，使用默认值就够了。

　　除了这些输入参数外，有两个PCA类的成员值得关注。

第一个是**explained_variance_**，它代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分。

第二个是**explained_variance_ratio_**，它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。

### 2.3. PCA实例

　　下面我们用一个实例来学习下scikit-learn中的PCA类使用。为了方便的可视化让大家有一个直观的认识，我们这里使用了三维的数据来降维。

　　首先我们生成随机数据并可视化，代码如下：

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets.samples_generator import make_blobs
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇
X, y = make_blobs(n_samples=10000, n_features=3, centers=[[3,3, 3], [0,0,0], [1,1,1], [2,2,2]], cluster_std=[0.2, 0.1, 0.2, 0.2], 
                  random_state =9)
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o')
plt.show()
```

 三维数据的分布图如下：

![](images/1042406-20170102203715206-1501576889.png)

　　我们先不降维，只对数据进行投影，看看投影后的三个维度的方差分布，代码如下：

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
```

　　输出如下：

[ 0.98318212 0.00850037 0.00831751]
[ 3.78483785 0.03272285 0.03201892]

　　可以看出投影后三个特征维度的方差比例大约为98.3%：0.8%：0.8%。投影后第一个特征占了绝大多数的主成分比例。

　　现在我们来进行降维，从三维降到2维，代码如下：

```python
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
```

　　输出如下：

[ 0.98318212 0.00850037]
[ 3.78483785 0.03272285]

　　这个结果其实可以预料，因为上面三个投影后的特征维度的方差分别为：[ 3.78483785 0.03272285 0.03201892]，投影到二维后选择的肯定是前两个特征，而抛弃第三个特征。

　　为了有个直观的认识，我们看看此时转化后的数据分布，代码如下：

```python
X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1],marker='o')
plt.show()
```

　　输出的图如下：

![](images/1042406-20170102204623581-74947365.png)

　　可见降维后的数据依然可以很清楚的看到我们之前三维图中的4个簇。

　　现在我们看看不直接指定降维的维度，而指定降维后的主成分方差和比例。

```python
pca = PCA(n_components=0.95)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)
```

　　我们指定了主成分至少占95%，输出如下：

```
[ 0.98318212]
[ 3.78483785]
1
```

　　可见只有第一个投影特征被保留。这也很好理解，我们的第一个主成分占投影特征的方差比例高达98%。只选择这一个特征维度便可以满足95%的阈值。我们现在选择阈值99%看看，代码如下：

```python
pca = PCA(n_components=0.99)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)
```

　　此时的输出如下：

```
[ 0.98318212  0.00850037]
[ 3.78483785  0.03272285]
2
```

　　这个结果也很好理解，因为我们第一个主成分占了98.3%的方差比例，第二个主成分占了0.8%的方差比例，两者一起可以满足我们的阈值。

　　最后我们看看让MLE算法自己选择降维维度的效果，代码如下：

```python
pca = PCA(n_components='mle')
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)
```

　　输出结果如下：

[ 0.98318212]
[ 3.78483785]
1

　　可见由于我们的数据的第一个投影特征的方差占比高达98.3%，MLE算法只保留了我们的第一个特征。

### 2.4 IncrementalPCA 

PCA 对象非常有用, 但对大型数据集有一定的限制。 最大的限制是 PCA 仅支持批处理，这意味着所有要处理的数据必须适合主内存。 IncrementalPCA 对象使用不同的处理形式使之允许部分计算：

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, IncrementalPCA

iris = load_iris()
X = iris.data
y = iris.target
n_components = 2

ipca = IncrementalPCA(n_components=n_components, batch_size=10)
X_ipca = ipca.fit_transform(X)

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

colors = ['navy', 'turquoise', 'darkorange']
for X_transformed, title in [(X_ipca, "Incremental PCA"), (X_pca, "PCA")]:
    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
        plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1], color=color, lw=2, label=target_name)
    if "Incremental" in title:
        err = np.abs(np.abs(X_pca) - np.abs(X_ipca)).mean()
    plt.title(title + " of iris dataset\nMean absolute unsigned error %.6f" % err)
else:
    plt.title(title + " of iris dataset")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.axis([-4, 4, -1.5, 1.5])
plt.show()
```



## 3. 因子分析

从变量中提取共性因子。

 因子分析要求原有变量间具有较强的相关性，否则，因子分析无法提取变量间的共性特征，如果相关系数小于0.3，则变量间的共线性较小，不适合因子分析；因子分析得到因子和原变量的关系，因此能够对因子进行解释。

```python
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import FactorAnalysis

iris=datasets.load_iris()
x=iris.data

fa=FactorAnalysis(n_components=3)  #指定3个因子作为新变量
fa.fit(x)
tran_x=fa.transform(x)
print(tran_x[:10])   

#输出新因子[[ -1.31686745e+00   5.97520018e-01   1.11217811e-03] [ -1.32710651e+00  -3.28650685e-01  -3.88642995e-01] [ -1.39619171e+00  -2.84417674e-01   1.31593688e-01] [ -1.33067991e+00  -5.98068260e-01   1.15421214e-03] [ -1.33261365e+00   6.06424839e-01   2.31749197e-01] [ -1.11358590e+00   1.37362223e+00   5.32582144e-01] [ -1.34512235e+00  -1.98028671e-01   6.99463537e-01] [ -1.28894846e+00   3.10292800e-01  -7.03646521e-02] [ -1.38762034e+00  -1.09325259e+00   3.50884895e-02] [ -1.32480883e+00  -1.89272934e-01  -6.00649914e-01]]
```

## 3 参考资料

1. [主成分分析（PCA）原理总结](https://www.cnblogs.com/pinard/p/6239403.html)
2. https://github.com/jakevdp/sklearn_tutorial/blob/master/notebooks/04.1-Dimensionality-PCA.ipynb

## 4 练习
1. https://www.jianshu.com/p/4298934e4a8e

