
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


plt.style.use('ggplot')


iris = datasets.load_iris()
print(type(iris))
print(iris.keys())

iris.data.shape
iris.target_names

x = iris.data
y = iris.target
df = pd.DataFrame(x, columns=iris.feature_names)
print(df.head())

#plotting with pandas' scatter_matrix
_ = pd.plotting.scatter_matrix(df, c=y, figsize=[8,8],s=150,marker='D')

