import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import tree
import math

class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.base_estimator=base_estimator
        self.n_estimators=n_estimators
        self.features=[]
        self.labels=[]
        self.models=[]
    def split(self,X,y,ratio):
        #selecting random dataset
        X['label']=y
        df=pd.DataFrame()
        size_df=round(len(X)*ratio)
        #selecting random rows with replacement till the dataset
        #size is not equal to ori dataset as ratio is kept 1
        while(len(df)<size_df):
            row_index=random.randrange(len(X))
            df=df.append(X[row_index:row_index+1])
        X.pop('label')
        X_train=df[df.columns[:-1]]
        y_train=df[df.columns[-1]]
        return X_train,y_train
    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.samples=X
        self.value=y
        
        for i in range(self.n_estimators):
            d_tree=tree.DecisionTreeClassifier()
            X_train,y_train=self.split(X,y,1)
            X_train=X_train.reset_index(drop=True)
            y_train=y_train.reset_index(drop=True)
            d_tree.fit(X_train,y_train)
            self.features.append(X_train)
            self.labels.append(y_train)
            self.models.append(d_tree)

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        pred=[]
        for model in self.models:
            pred.append(np.array(model.predict(X)))
        pred=np.array(pred)
        pred_t=pred.T
        predictions=[]
        
        #Taking mode of all estimators for classification
        for i in range(len(pred_t)):
            predictions.append(np.bincount(pred_t[i]).argmax())
        return pd.Series(predictions)



    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        print("Printing decision surfaces of individual estimator ")
        plot_colors = "rb"
        plot_step = 0.02
        n_classes = 2
        
        for k in range (self.n_estimators):
            clf=self.models[k]
            X=np.array(self.features[k])
            Y=np.array(self.labels[k])
            plt.subplot(1, self.n_estimators, k+1 )
            x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
            y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
            for i, color in zip(range(n_classes), plot_colors):
                idx = np.where(Y == i)
                plt.scatter(X[idx,0], X[idx,1],c=color,cmap=plt.cm.RdBu, edgecolor='black', s=15) 
        plt.suptitle("Bagging:Decision surface of a decision tree using two features")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")
        plt.show()
        fig1=plt

        print("Printing decision surface of combined estimator ")
        plot_colors = "rb"
        plot_step = 0.02
        n_classes = 2
        
        
        X=np.array(self.samples)
        Y=np.array(self.value)
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
        y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = self.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()]))
        Z=np.array(Z)
        Z = Z.reshape(xx.shape)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(Y == i)
            plt.scatter(X[idx,0], X[idx,1],c=color,cmap=plt.cm.RdBu, edgecolor='black', s=15) 
        plt.suptitle("Bagging:Combined decision surface using 2 features")
        #plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")
        plt.show()
        fig2=plt

        return [fig1,fig2]



