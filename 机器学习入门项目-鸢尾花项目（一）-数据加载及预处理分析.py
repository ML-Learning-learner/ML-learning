# 导入所需要的库
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 数据加载
data = load_iris()
# 获取标签
iris_target = data.target
# 数据转换为DataFrame格式，方便后续操作
iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)
# 查看数据
# print(iris_features)
# 查看前5行数据
print(iris_features.head(5))
# 查看数据长度
# print(len(iris_features))

# 花萼长度的四种图
# 创建2x2的子图布局
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(12, 8))
# 第一个子图：直方图
sns.histplot(data = iris_features, x ='sepal length (cm)', kde = True, ax = axes[0, 0])
axes[0, 0].set_title('花萼长度直方图')
# 第二个子图：箱线图
sns.boxplot(data = iris_features, x ='sepal length (cm)', ax = axes[0, 1])
axes[0, 1].set_title('花萼长度箱线图')
# 第三个子图：小提琴图
sns.violinplot(data = iris_features, x ='sepal length (cm)', ax = axes[1, 0])
axes[1, 0].set_title('花萼长度小提琴图')
# 第四个子图：核密度估计图
sns.kdeplot(data = iris_features['sepal length (cm)'], fill = True, ax = axes[1, 1])
axes[1, 1].set_title('花萼长度核密度估计图')
# 调整子图之间的间距
plt.tight_layout()
# 显示图表
plt.show()

# 创建2x2的子图布局
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(12, 8))
# 第一个子图：直方图
sns.histplot(data = iris_features, x ='sepal length (cm)', kde = True, ax = axes[0, 0])
axes[0, 0].set_title('花萼宽度直方图')
# 第二个子图：箱线图
sns.boxplot(data = iris_features, x ='sepal length (cm)', ax = axes[0, 1])
axes[0, 1].set_title('花萼宽度箱线图')
# 第三个子图：小提琴图
sns.violinplot(data = iris_features, x ='sepal length (cm)', ax = axes[1, 0])
axes[1, 0].set_title('花萼宽度小提琴图')
# 第四个子图：核密度估计图
sns.kdeplot(data = iris_features['sepal length (cm)'], fill = True, ax = axes[1, 1])
axes[1, 1].set_title('花萼宽度核密度估计图')
# 调整子图之间的间距
plt.tight_layout()
# 显示图表
plt.show()


# 创建2x2的子图布局
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(12, 8))
# 第一个子图：直方图
sns.histplot(data = iris_features, x ='petal length (cm)', kde = True, ax = axes[0, 0])
axes[0, 0].set_title('花瓣长度直方图')
# 第二个子图：箱线图
sns.boxplot(data = iris_features, x ='petal length (cm)', ax = axes[0, 1])
axes[0, 1].set_title('花瓣长度箱线图')
# 第三个子图：小提琴图
sns.violinplot(data = iris_features, x ='petal length (cm)', ax = axes[1, 0])
axes[1, 0].set_title('花瓣长度小提琴图')
# 第四个子图：核密度估计图
sns.kdeplot(data = iris_features['petal length (cm)'], fill = True, ax = axes[1, 1])
axes[1, 1].set_title('花瓣长度核密度估计图')
# 调整子图之间的间距
plt.tight_layout()
# 显示图表
plt.show()


# 创建2x2的子图布局
fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(12, 8))
# 第一个子图：直方图
sns.histplot(data = iris_features, x ='petal width (cm)', kde = True, ax = axes[0, 0])
axes[0, 0].set_title('花瓣宽度直方图')
# 第二个子图：箱线图
sns.boxplot(data = iris_features, x ='petal width (cm)', ax = axes[0, 1])
axes[0, 1].set_title('花瓣宽度箱线图')
# 第三个子图：小提琴图
sns.violinplot(data = iris_features, x ='petal width (cm)', ax = axes[1, 0])
axes[1, 0].set_title('花瓣宽度小提琴图')
# 第四个子图：核密度估计图
sns.kdeplot(data = iris_features['petal width (cm)'], fill = True, ax = axes[1, 1])
axes[1, 1].set_title('花瓣宽度核密度估计图')
# 调整子图之间的间距
plt.tight_layout()
# 显示图表
plt.show()
