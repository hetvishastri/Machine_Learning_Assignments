from .base import DecisionTree
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import tree
import math

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        
        self.n_estimators=n_estimators
        self.criterion=criterion
        self.max_depth=max_depth
        self.feat=[]
        self.models=[]
        self.data=[]
        self.labels=[]

    def split(self,X,y):

        features=[]
        df=pd.DataFrame()
        #selecting random features for training
        size_df=random.randrange(1,X.shape[1]+1)
        while(len(features)<size_df):
            col_index=random.randrange(X.shape[1])
            if(col_index not in features):
                df[len(features)]=X[col_index]
                features.append(col_index)
            
        X_train=df
        return X_train,features


    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.samples=X
        self.value=y
        for i in range(self.n_estimators):
            d_tree=tree.DecisionTreeClassifier(max_depth=5)
            X_train,features=self.split(X,y)

            #if part 1 of question 7
            # if(self.criterion=="gini_index" or self.criterion=="information_gain"):
            #     X_train,features=self.split(X,y)
            # #for part 2 of question 7
            # # else:
            # #     X_train,features=X,[1,3]
            # else:
            #     X_train,features=self.split(X,y)
            X_train=X_train.reset_index(drop=True)
            d_tree.fit(X_train,y)
            self.feat.append(features)
            self.models.append(d_tree)
            self.data.append(X_train)
            self.labels.append(y)



        

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pred=[]
        
        for i in range(len(self.feat)):
            pred.append(self.models[i].predict(X[self.feat[i]]))
            # if(self.criterion=="gini_index" or self.criterion=="information_gain"):
            #     pred.append(self.models[i].predict(X[self.feat[i]]))
            # # else:
            # #     pred.append(self.models[i].predict(X))
            # else:
            #     pred.append(self.models[i].predict(X[self.feat[i]]))

            
        pred=np.array(pred)
        pred_t=pred.T
        predictions=[]
        #selecting mode as it is classification
        for i in range(len(pred_t)):
            predictions.append(np.bincount(pred_t[i]).argmax())
        return pd.Series(predictions)

        

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimators

        3. Creates a figure showing the combined decision surface

        """
        
        print("Printing decision tree")
        for k in range(int(self.n_estimators)):
            if(len(self.feat[k])==1):
                if(self.feat[0]==0):
                    features=["Sepal width"]
                else:
                    features=["Petal width"]
            else:
                features=["Sepal width","Petal width"]
                
            tree.plot_tree(self.models[k],feature_names=features,class_names=['Iris-setosa','Iris-versicolor','Iris-virginica'],filled=True,proportion=True,fontsize=5)
            plt.show()
        

        print("Printing decision surfaces ")
        plot_colors = "ryb"
        plot_step = 0.02
        n_classes = 3
        
        for k in range (self.n_estimators):
            clf=self.models[k]
            X=np.array(self.data[k])
            Y=np.array(self.labels[k])
            plt.subplot(2, 5, k+1 )
            x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
            if(X.shape[1]==2):
                y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
                plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
                Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
                plt.xlabel('Sepal width')
                plt.ylabel('Petal width')
            else:
                xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(x_min, x_max, plot_step))
                plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
                Z = clf.predict(np.c_[xx.ravel()])
                Z = Z.reshape(xx.shape)
                cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
                if(self.feat[k][0]==0):
                    plt.xlabel('Sepal width')
                else:
                    plt.xlabel('Petal width')
                
            features=['Iris-setosa','Iris-versicolor','Iris-virginica']
            for i, color in zip(range(n_classes), plot_colors):
                idx = np.where(Y == i)
                if(X.shape[1]==2):
                    plt.scatter(X[idx,0], X[idx,1],c=color,label=features[i],cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
                else:
                     plt.scatter(X[idx,0],X[idx,0],c=color,label=features[i],cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
        plt.suptitle("Random_forest:Decision surface of a decision tree using two features")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")
        plt.show()
        fig1=plt

        print("Printing decision surface of combined estimator ")
        plot_colors = "ryb"
        plot_step = 0.02
        n_classes = 3    
        X=np.array(self.samples)
        Y=np.array(self.value)
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
        y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = self.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()]))
        Z=np.array(Z)
        Z = Z.reshape(xx.shape)
        plt.xlabel('Sepal width')
        plt.ylabel('Petal width') 
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
        features=['Iris-setosa','Iris-versicolor','Iris-virginica']
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(Y == i)
            plt.scatter(X[idx,0], X[idx,1],c=color,label=features[i],cmap=plt.cm.RdYlBu, edgecolor='black', s=15) 
        plt.suptitle("Randomforest:Combined decision surface")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")
        plt.show()
        
        fig2=plt

        return [fig1,fig2]



class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.feat=[]
        self.models=[]
        
    def split(self,X,y):
        features=[]
        df=pd.DataFrame()
        size_df=2
        
        while(len(features)<size_df):
            col_index=random.randrange(X.shape[1])
            df[len(features)]=X[col_index]
            features.append(col_index)

        X_train=df
        return X_train,features

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        for i in range(self.n_estimators):
            d_tree=tree.DecisionTreeRegressor()
            X_train,features=self.split(X,y)
            X_train=X_train.reset_index(drop=True)
            d_tree.fit(X_train,y)
            self.feat.append(features)
            self.models.append(d_tree)

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pred=[]
        for i in range(len(self.feat)):
            pred.append(self.models[i].predict(X[self.feat[i]]))
        pred=np.array(pred)
        pred_t=pred.T
        predictions=[]
        for i in range(len(pred_t)):
            predictions.append(np.mean(pred_t[i]))

        return pd.Series(predictions)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        pass
