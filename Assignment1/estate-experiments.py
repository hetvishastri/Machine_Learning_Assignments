
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split
from metrics import *

np.random.seed(42)

# Read real-estate data set
# ...
# 
real_estate_data=pd.read_excel("Real estate valuation data set.xlsx",names=[0,1,2,3,4,5,6,"label"])
X=real_estate_data[real_estate_data.columns[:-1]]
y=real_estate_data[real_estate_data.columns[-1]]

X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(X, y, test_size=0.3)

skcit_tree=DecisionTreeRegressor()
skcit_tree=skcit_tree.fit(X_train_sk,y_train_sk)
y_hat_sk=skcit_tree.predict(X_test_sk)

print("This is scikit learn tree accuracy")

print('RMSE: ', rmse(y_hat_sk, y_test_sk))
print('MAE: ', mae(y_hat_sk, y_test_sk))


real_estate_data=pd.read_excel("Real estate valuation data set.xlsx",names=[0,1,2,3,4,5,6,"label"])
X=real_estate_data[real_estate_data.columns[:-1]]
y=real_estate_data[real_estate_data.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test=X_test.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)
X_train=X_train.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)

tree = DecisionTree(criterion="information_gain") #Split based on Inf. Gain
tree.fit(X_train, y_train)
#tree.plot()
y_hat_my = tree.predict(X_test)
print("This is my tree accuracy")
print('RMSE: ', rmse(y_hat_my, y_test))
print('MAE: ', mae(y_hat_my, y_test))

