
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6)

# Fit classifier to the data
knn.fit(iris['data'],iris['target'])  


# Prediction for the training data
pred=knn.predict(iris['data'])
print('Prediction{}'.format(pred))


# Prediction for a set of random data
x_new = np.random.randint(10, size=(6,4))
pred_new = knn.predict(x_new)
print('Prediction{}'.format(pred_new))




