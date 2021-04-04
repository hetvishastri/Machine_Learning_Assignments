import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from sklearn.model_selection import train_test_split
from metrics import *

np.random.seed(42)

print("Decision tree on iris dataset")

# Read IRIS data set
iris=pd.read_csv("iris.csv",names=[0,1,2,3,"label"])
#Used some function in base.py which works on numeric feature
#converted into classes
iris['label'] = pd.factorize(iris['label'])[0]
#sampling
iris.sample(frac=1)

X=iris[iris.columns[:-1]]
y=iris[iris.columns[-1]].astype('category')


# ...
#Splitting dataset into 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test=X_test.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)
X_train=X_train.reset_index(drop=True)
y_train=y_train.reset_index(drop=True)

# Checking accuracy of my decision tree on iris dataset
tree = DecisionTree(criterion="information_gain") #Split based on Inf. Gain
tree.fit(X_train, y_train)
tree.plot()
y_hat = tree.predict(X_test)
print('Criteria :', "information gain")
print('Accuracy: ', accuracy(y_hat, y_test))
#per class precision and recall
for cls in y.unique():
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))


print("5 fold cross validation")
#5 fold cross validation
def five_fold_cross_validation(dataset,i,k_fold):
    n=len(dataset)//k_fold
    #selecting part of dataset
    test=dataset[n*i:n*(i+1)]
    test=test.reset_index(drop=True)
    #when first and last part of dataset are test set
    if(i==0):
        train=dataset[n*i+1:]
    elif(i==(k_fold-1)):
        train=dataset[:n*i]
    else:
        train_left=dataset[:n*i]
        train_right=dataset[n*(i+1):]
        train=pd.concat([train_left,train_right],axis=0)
    train=train.reset_index(drop=True)
    return train,test

avg=0
#initializing the tree
tree_cross=DecisionTree("information_gain")  
k_fold=5#number of folds
for i in range(k_fold):
    train,test=five_fold_cross_validation(iris,i,k_fold)
    X_train=train[train.columns[:-1]]
    y_train=train[train.columns[-1]].astype('category')
    X_test=test[test.columns[:-1]]
    y_test=test[test.columns[-1]].astype('category')
    tree_cross.fit(X_train,y_train)
    y_hat=tree_cross.predict(X_test)
    acc=accuracy(y_hat,y_test)
    print("-----------")
    print("Accuracy of",i+1,"th fold:",acc)
    avg=avg+acc
average_accuaracy=avg/k_fold
print("This is the average accuracy",average_accuaracy)

print("Nested cross validation")
for i in range(k_fold):
    #cross validation for splitting val+train and test set
    trainplusval,test=five_fold_cross_validation(iris,i,k_fold)
    X_test=test[test.columns[:-1]]
    y_test=test[test.columns[-1]].astype('category')
    max_accuracy=0
    best_tree=None
    optinum_depth=0
    #iterating for 1-6 depth size
    for depth in range(7):
        avg=0
        for j in range(k_fold):
             #cross validation for splitting train and validation set
            train,val_data=five_fold_cross_validation(trainplusval,j,k_fold)
            X_train=train[train.columns[:-1]]
            y_train=train[train.columns[-1]].astype('category')
            #validation set
            X_val_data=val_data[val_data.columns[:-1]]
            y_val_data=val_data[val_data.columns[-1]].astype('category')
            tree_cross=DecisionTree("information_gain",max_depth=depth) 
            tree_cross.fit(X_train,y_train)
            y_val_hat=tree_cross.predict(X_val_data)
            #checking accuracy on validation set
            acc=accuracy(y_val_hat,y_val_data)
            avg=avg+acc
        average_accuaracy=avg/(k_fold-1)
        if(average_accuaracy>max_accuracy):
            max_accuracy=average_accuaracy
            optinum_depth=depth
            best_tree=tree_cross
    #predicting on best model with optimum depth
    y_hat=best_tree.predict(X_test)
    acc_final=accuracy(y_hat,y_test)
    print("-------")
    print("Accuracy of",i+1,"th fold:",acc_final,"Optimum depth is",optinum_depth)



