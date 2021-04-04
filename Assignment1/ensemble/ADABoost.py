
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import tree
import math
class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators=3): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.base_estimator=base_estimator
        self.n_estimators=n_estimators
        self.features=[]
        self.labels=[]
        self.models=[]

    def compute_alpha(self,X,y,d_tree,weights):
        
        num_sample=len(X)
        error=0
        #Calculating error
        for i in range(num_sample):
            if(d_tree.predict(X.iloc[[i]])!=y[i]):
                error+=weights[i]    
        alpha=0.5*np.log2((1-error)/error)
        #Assigning new weights
        for i in range(num_sample):
            if(d_tree.predict(X.iloc[[i]])==y[i]):
                weights[i]=weights[i]*np.exp(-alpha)
            else:
                weights[i]=weights[i]*np.exp(-alpha)
        #Normalizing weights
        for i in range(len(weights)):
            weights[i]=weights[i]/sum(weights)
        return weights,alpha,error

    def fit(self, X, y):
        self.samples=X
        self.value=y
        num_sample=len(X)
        self.clf=np.unique(y)
        weights=[]
        self.w=[]
        self.alphas=[]
        errors=[]
        #Initializing weights
        for i in range(num_sample):
            weights.append(1/num_sample)
        self.w.append(weights)
        for i in range(self.n_estimators):
            d_tree=tree.DecisionTreeClassifier(max_depth=1)
            d_tree.fit(X,y,sample_weight=weights)
            self.models.append(d_tree)
            self.features.append(X)
            self.labels.append(y)
            weights,alpha,error=self.compute_alpha(X,y,d_tree,weights)
            errors.append(error)
            self.alphas.append(alpha)
            self.w.append(weights)
            


        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        

    def predict(self, X):
        
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hat=[]
        for i in range(len(X)):
            final_pred=0
            for j in range(self.n_estimators):
                #If class 1 then prediction is 1
                if(self.models[j].predict(X.iloc[[i]])==self.clf[0]):
                    pred=1
                #If class 2 then prediction is -1
                else:
                    pred=-1
                final_pred=final_pred+self.alphas[j]*(pred)
            #If prediction is positive then class 1
            if(final_pred>0):
                y_hat.append(self.clf[0])
            #If prediction is negative then class 2
            else:
                y_hat.append(self.clf[1])
        return pd.Series(y_hat)


        

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

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
            weight=np.array(self.w[k])
            plt.subplot(1, self.n_estimators, k+1 )
            x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
            y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.xlabel('Sepal width')
            plt.ylabel('Petal width')
            cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
            features=["Virginica","Non Virginica"]
            #features=["0","1"]
            for i, color in zip(range(n_classes), plot_colors):
                idx = np.where(Y == i)
                plt.scatter(X[idx,0], X[idx,1],c=color,label=features[i],cmap=plt.cm.RdBu, edgecolor='black', s=15) 
            plt.title("alpha="+str(self.alphas[k]))
        plt.suptitle("ADABoost:Decision surface of a decision tree using two features")
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
        weight=np.array(self.w[k])
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
        y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),np.arange(y_min, y_max, plot_step))
        plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
        Z = self.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()]))
        Z=np.array(Z)
        Z = Z.reshape(xx.shape)
        plt.xlabel('Sepal width')
        plt.ylabel('Petal width')
        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu)
        features=["Virginica","Non Virginica"]
        #features=["0","1"]
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(Y == i)
            plt.scatter(X[idx,0], X[idx,1],c=color,label=features[i],cmap=plt.cm.RdBu, edgecolor='black', s=15) 
        plt.suptitle("ADABoost:Combined decision surface")
        plt.legend(loc='lower right', borderpad=0, handletextpad=0)
        plt.axis("tight")
        plt.show()
        
        fig2=plt

        return [fig1,fig2]

