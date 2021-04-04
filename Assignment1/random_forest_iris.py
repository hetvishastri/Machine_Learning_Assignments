import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

###Write code here
np.random.seed(42)

# Read IRIS data set
iris=pd.read_csv("iris.csv",names=[0,1,2,3,"label"])
iris['label'] = pd.factorize(iris['label'])[0]
#sampling
iris.sample(frac=1)

X1=iris[iris.columns[1:2]]
X1=X1.reset_index(drop="True")
X2=iris[iris.columns[3:4]]
X2=X2.reset_index(drop="True")
X=pd.concat((X1,X2),axis=1,ignore_index=True)
y=iris[iris.columns[-1]].astype('category')


# ...

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
X_train=X_train.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)
X_test=X_test.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)


Classifier_RF = RandomForestClassifier(10)
Classifier_RF.fit(X_train, y_train)
y_hat = Classifier_RF.predict(X_test)
#Classifier_RF.plot()
print('Criteria :', 'gini_index')
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))
