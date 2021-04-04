"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
#from linearRegression.linearRegression import LinearRegression
from sklearn.model_selection import train_test_split
warnings. filterwarnings("ignore")

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
tree = DecisionTree(criterion=criteria)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
#[fig1, fig2] = Classifier_AB.plot()
plt.savefig("RIDO_M_fit.png")
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))


# print("This is iris dataset")
# ##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features
np.random.seed(42)



# Read IRIS data set
iris=pd.read_csv("iris.csv",names=[0,1,2,3,"label"])
iris.label[iris.label=="Iris-setosa"]=0
iris.label[iris.label=="Iris-versicolor"]=0
iris.label[iris.label=="Iris-virginica"]=1
iris.sample(frac=1.0)
# Read IRIS data set
# iris=pd.read_csv("iris.csv",names=[0,1,2,3,"label"])
# iris['label'] = pd.factorize(iris['label'])[0]


print("ADAboost on iris dataset")
X1=iris[iris.columns[1:2]]
X1=X1.reset_index(drop="True")
X2=iris[iris.columns[3:4]]
X2=X2.reset_index(drop="True")
X=pd.concat((X1,X2),axis=1)
y=iris[iris.columns[-1]].astype('category')


# ...

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
X_train=X_train.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)
X_test=X_test.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)

criteria = 'information_gain'
tree = DecisionTree(criterion=criteria)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X_train, y_train)
y_hat = Classifier_AB.predict(X_test)
#[fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))


####For comparing sklear decision tree with depth 1 

# from sklearn import tree
# tree_nor=tree.DecisionTreeClassifier(max_depth=1)
# tree_nor.fit(X_train, y_train)
# y_hat_nor = tree_nor.predict(X_test)
# y_hat_nor=pd.Series(y_hat_nor)
# print('Accuracy: ', accuracy(y_hat_nor, y_test))
# for cls in y.unique():
#      print('Precision: ', precision(y_hat_nor, y_test, cls))
#      print('Recall: ', recall(y_hat_nor, y_test, cls))

