# 数据预处理
<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%201.jpg">
</p>

### 步骤一 导入 numpy 和 pandas 的库


```python
import numpy as np
import pandas as pd
```

### 步骤二 导入 csv 数据集


```python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
```


```python
print(X,Y, sep='\n\n')
```

    [['France' 44.0 72000.0]
     ['Spain' 27.0 48000.0]
     ['Germany' 30.0 54000.0]
     ['Spain' 38.0 61000.0]
     ['Germany' 40.0 nan]
     ['France' 35.0 58000.0]
     ['Spain' nan 52000.0]
     ['France' 48.0 79000.0]
     ['Germany' 50.0 83000.0]
     ['France' 37.0 67000.0]]
    
    ['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']
    

可以看到成功导入了数据集, X 获取数据集中所有行除了最后一列的数据, 而 Y 则是数据集中最后一列的数据

### 步骤三 处理缺失的数据


```python
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
```


```python
print(X)
```

    [['France' 44.0 72000.0]
     ['Spain' 27.0 48000.0]
     ['Germany' 30.0 54000.0]
     ['Spain' 38.0 61000.0]
     ['Germany' 40.0 63777.77777777778]
     ['France' 35.0 58000.0]
     ['Spain' 38.77777777777778 52000.0]
     ['France' 48.0 79000.0]
     ['Germany' 50.0 83000.0]
     ['France' 37.0 67000.0]]
    

可以看到, X 中 1 和 2 列中的缺失数据使用该列的均值来填充, `strategy=mean` 表示策略是平均值, `axis=0` 表示按列.

### 步骤四 编码分类数据


```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
```


```python
print(X)
```

    [[0 44.0 72000.0]
     [2 27.0 48000.0]
     [1 30.0 54000.0]
     [2 38.0 61000.0]
     [1 40.0 63777.77777777778]
     [0 35.0 58000.0]
     [2 38.77777777777778 52000.0]
     [0 48.0 79000.0]
     [1 50.0 83000.0]
     [0 37.0 67000.0]]
    

可以看到, 第一列的字符串被转换成了标签值


```python
onehotencoder = OneHotEncoder(categorical_features=[0], n_values=[3])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()

Y = labelencoder_Y.fit_transform(Y) # 同样地将 'Yes' 和 'No' 转为标签值
```


```python
print(X, Y, sep='\n\n')
```

    [[1.00000000e+00 0.00000000e+00 0.00000000e+00 4.40000000e+01
      7.20000000e+04]
     [0.00000000e+00 0.00000000e+00 1.00000000e+00 2.70000000e+01
      4.80000000e+04]
     [0.00000000e+00 1.00000000e+00 0.00000000e+00 3.00000000e+01
      5.40000000e+04]
     [0.00000000e+00 0.00000000e+00 1.00000000e+00 3.80000000e+01
      6.10000000e+04]
     [0.00000000e+00 1.00000000e+00 0.00000000e+00 4.00000000e+01
      6.37777778e+04]
     [1.00000000e+00 0.00000000e+00 0.00000000e+00 3.50000000e+01
      5.80000000e+04]
     [0.00000000e+00 0.00000000e+00 1.00000000e+00 3.87777778e+01
      5.20000000e+04]
     [1.00000000e+00 0.00000000e+00 0.00000000e+00 4.80000000e+01
      7.90000000e+04]
     [0.00000000e+00 1.00000000e+00 0.00000000e+00 5.00000000e+01
      8.30000000e+04]
     [1.00000000e+00 0.00000000e+00 0.00000000e+00 3.70000000e+01
      6.70000000e+04]]
    
    [0 1 0 0 1 1 0 1 0 1]
    

使用 OneHotEncoder (独热编码) 转换之后的数据

其中 `categorical_features` 是需要独热编码的列索引，`n_values` 是对应 categorical_features 中各列下类别的数目

例如: 这里的 X 第一列使用 OneHotEncoder 将 0, 1, 2 分别转换成 [1,0,0], [0,1,0] 和 [0,0,1].

具体为啥要转换呢?

### 步骤五 将数据集分割为训练集和测试集


```python
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
```


```python
print(X_train, X_test, Y_train, Y_test, sep='\n\n')
```

    [[0.00000000e+00 1.00000000e+00 0.00000000e+00 4.00000000e+01
      6.37777778e+04]
     [1.00000000e+00 0.00000000e+00 0.00000000e+00 3.70000000e+01
      6.70000000e+04]
     [0.00000000e+00 0.00000000e+00 1.00000000e+00 2.70000000e+01
      4.80000000e+04]
     [0.00000000e+00 0.00000000e+00 1.00000000e+00 3.87777778e+01
      5.20000000e+04]
     [1.00000000e+00 0.00000000e+00 0.00000000e+00 4.80000000e+01
      7.90000000e+04]
     [0.00000000e+00 0.00000000e+00 1.00000000e+00 3.80000000e+01
      6.10000000e+04]
     [1.00000000e+00 0.00000000e+00 0.00000000e+00 4.40000000e+01
      7.20000000e+04]
     [1.00000000e+00 0.00000000e+00 0.00000000e+00 3.50000000e+01
      5.80000000e+04]]
    
    [[0.0e+00 1.0e+00 0.0e+00 3.0e+01 5.4e+04]
     [0.0e+00 1.0e+00 0.0e+00 5.0e+01 8.3e+04]]
    
    [1 1 1 0 1 0 0 1]
    
    [0 0]
    

数据集以 8:2 的比例将 X 和 Y 分割出训练集和测试集

`test_size` 表示测试样本的占比

`random_state` 是随机数的种子

### 步骤六 特征缩放


```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
```


```python
print(X_train, X_test, sep='\n\n')
```

    [[-1.          2.64575131 -0.77459667  0.26306757  0.12381479]
     [ 1.         -0.37796447 -0.77459667 -0.25350148  0.46175632]
     [-1.         -0.37796447  1.29099445 -1.97539832 -1.53093341]
     [-1.         -0.37796447  1.29099445  0.05261351 -1.11141978]
     [ 1.         -0.37796447 -0.77459667  1.64058505  1.7202972 ]
     [-1.         -0.37796447  1.29099445 -0.0813118  -0.16751412]
     [ 1.         -0.37796447 -0.77459667  0.95182631  0.98614835]
     [ 1.         -0.37796447 -0.77459667 -0.59788085 -0.48214934]]
    
    [[ 0.  0.  0. -1. -1.]
     [ 0.  0.  0.  1.  1.]]
    

emmmm..最终我获得了这堆看起来很奇怪的东西..而且Day 1就这样结束了...

StandardScaler 作用：去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本
