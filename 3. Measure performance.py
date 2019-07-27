
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()


from sklearn.model_selection import train_test_split

x= iris['data']  #features
y= iris['target'] #target

#Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.3,
                                                    random_state=21,
                                                    stratify=y)

knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print('Test set predictions:\n{}'.format(y_pred))

knn.score(x_test, y_test)


#
#Underfitting and overfitting
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    
    train_accuracy[i] = knn.score(x_train, y_train)
    test_accuracy[i] = knn.score(x_test, y_test)

plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()





