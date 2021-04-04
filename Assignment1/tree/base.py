"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index,information_gain_dr,gini_score

np.random.seed(42)

class DecisionTree():
    def __init__(self, criterion, max_depth=6):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        self.max_depth=max_depth
        self.criterion=criterion


    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        input=X.dtypes[0].name
        output=y.dtype.name
        depth=0
        attr=list(X.keys())
        if(input!="category" and output!="category"):
            #print("Real input real output")
            self.flag=1
            self.tree=self.real_real(X,y,depth)
        if(input!="category" and output=="category"):
            #print("Real input discrete output")
            self.flag=2
            self.tree=self.real_discrete(X,y,depth)
        if(input=="category" and output=="category"):
            #print("Discrete input Discrete output")
            self.flag=3
            self.tree=self.discrete_discrete(X,y,depth,attr,None)
        if(input=="category" and output!="category"):
            #print("Discrete input Real output")
            self.flag=4
            self.tree=self.discrete_real(X,y,depth,attr,None)
        

    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        #prediction for discrete input
        if(self.flag==3 or self.flag==4):
            y_hat=[]
            for i in range(len(X)):
                sub_tree = self.tree
                while type(sub_tree) == dict:
                    root = list(sub_tree.keys())[0]
                    value = X[root][i]
                    sub_tree = sub_tree[root][value]
                y_hat.append(sub_tree)
            return pd.Series(y_hat)
        #prediction for real input
        else:
            y_hat=[]
            for i in range(len(X)):
                sub_tree = self.tree
                while type(sub_tree) == dict:
                    root = list(sub_tree.keys())[0]
                    value = X[root][i]
                    threshold=float(list(sub_tree[root].keys())[0][1:])
                    if(value<threshold):
                      sub_tree=sub_tree[root][list(sub_tree[root].keys())[0]]
                    else:
                      sub_tree=sub_tree[root][list(sub_tree[root].keys())[1]]
                y_hat.append(sub_tree)
            return pd.Series(y_hat)

    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        from pprint import pprint
        pprint(self.tree)
    
    #Functions for four cases
        ####################################################################################################################
    #################################################### Real Input ################################################
    #Real input real output
    def real_real(self,X,y,depth):
      #getting optimum attribute and threshold to divide
      attr,th = self.get_split(X,y)
      left_x=X.where(X[attr]<th).dropna()
      left_y=y.where(X[attr]<th).dropna()
      right_x=X.where(X[attr]>=th).dropna()
      right_y=y.where(X[attr]>=th).dropna()
      left_x.reset_index(drop="True",inplace=True)
      left_y.reset_index(drop="True",inplace=True)
      right_x.reset_index(drop="True",inplace=True)
      right_y.reset_index(drop="True",inplace=True)
      #if any of the child is empty return with the mean
      if len(left_x)==0:
        return np.mean(right_y)
      if len(right_x)==0:
        return np.mean(left_y)
      node={attr:{}}
      #check the current depth else return with mean
      if depth >= self.max_depth:
        left_y.append(right_y)
        return np.mean(left_y)
      node[attr]['<'+str(th)]= self.real_real(left_x,left_y,depth+1) 
      node[attr]['>='+str(th)]= self.real_real(right_x,right_y,depth+1)
      self.tree=node
      return node

    #Real input discrete output
    def real_discrete(self,X,y,depth):
      #getting optimum attribute and threshold to divide
      attr,th = self.get_split(X,y)
      left_x=X.where(X[attr]<th).dropna()
      left_y=y.where(X[attr]<th).dropna()
      right_x=X.where(X[attr]>=th).dropna()
      right_y=y.where(X[attr]>=th).dropna()
      left_x.reset_index(drop="True",inplace=True)
      left_y.reset_index(drop="True",inplace=True)
      right_x.reset_index(drop="True",inplace=True)
      right_y.reset_index(drop="True",inplace=True)
      #if any of the child is empty return with the mode
      if len(left_x)==0:
        return self.max_occuring(right_y)
      if len(right_x)==0:
        return self.max_occuring(left_y)
      node={attr:{}}
      #check the current depth else return with mode
      if depth >= self.max_depth:
        left_y.append(right_y)
        return self.max_occuring(left_y)
      node[attr]['<'+str(th)]= self.real_discrete(left_x,left_y,depth+1) 
      node[attr]['>='+str(th)]= self.real_discrete(right_x,right_y,depth+1)
      self.tree=node
      return node

    #Functions for real input 
    def best_split(self,attr,y,cutoff):
      left_y=y.where(attr<cutoff).dropna()
      right_y=y.where(attr>=cutoff).dropna()
      return left_y,right_y


    def get_split(self,X,y):
      min_gini = np.inf
      min_info=np.inf
      #iterating for optimum attribute and threshold
      for key in X.keys():
        for i  in range(0,len(X)):
          left_y,right_y = self.best_split(X[key],y,X[key][i])
          if(self.criterion=="information_gain"):
              info_left=entropy(left_y)
              info_right=entropy(right_y)
              info=(len(left_y)/len(y))*info_left+(len(right_y)/len(y))*info_right
              if(info<=min_info):
                min_info=info
                optinum_attr=key
                threshold=(X[key][i])
          elif(self.criterion=="gini_index"):
              gini_left=gini_index(left_y)
              gini_right=gini_index(right_y)
              gini=(len(left_y)/len(y))*gini_left+(len(right_y)/len(y))*gini_right
              if(gini <= min_gini):
                min_gini=gini
                optinum_attr=key
                threshold=(X[key][i])
      return optinum_attr,threshold

    ####################################################################################################################
    #################################################### Discrete Input ################################################

    #Discrete input discrete output
    def discrete_discrete(self,X,y,depth,attr,parent):
      #if all are same then return with that class
      if(self.check_all_same(y)==1):
        return list(y.unique())[0]
      #check the current depth else return with mode
      elif(depth>=self.max_depth):
        return self.max_occuring(y)
      #if target attributes are over then return with mode in previous iteration
      elif(len(attr)==0):
        return parent
      else:
        if(self.criterion=="information_gain"):
          min_info=0
          for key in attr:
            info=information_gain(y,X[key])
            if(info>=min_info):
              min_info=info
              optinum_attr=key
        if(self.criterion=="gini_index"):
          min_gini=0
          for key in attr:
            gini=gini_score(y,X[key])
            if(gini>=min_gini):
              min_gini=gini
              optinum_attr=key
        #storing parent 
        parent=self.max_occuring(y)
        attr_val=np.unique(X[optinum_attr])
        node={}
        node[optinum_attr]={}
        attr=[i for i in attr if i!= optinum_attr]

        for val in attr_val:
          #extraxting the subdata of optimum atrribute
          attr_data=X.where(X[optinum_attr]==val).dropna()
          target_attr_data=y.where(X[optinum_attr]==val).dropna()
          node[optinum_attr][val]=self.discrete_discrete(attr_data,target_attr_data,depth+1,attr,parent)
        self.tree=node
        return node
      
    #Discrete input real output
    def discrete_real(self,X,y,depth,attr,parent):
      #check the current depth else return with mode
      if(depth>=self.max_depth):
        return (np.mean(y))
      #if target attributes are over then return with mode in previous iteration
      elif(len(attr)==0):
        return parent
      else:
        if(self.criterion=="information_gain"):
          min_info=0
          for key in attr:
            info=information_gain_dr(y,X[key])
            if(info>=min_info):
              min_info=info
              optinum_attr=key
        if(self.criterion=="gini_index"):
          min_gini=0
          for key in attr:
            gini=gini_score(y,X[key])
            if(gini>=min_gini):
              min_gini=gini
              optinum_attr=key
        #storing parent 
        parent=np.mean(y)
        attr_val=np.unique(X[optinum_attr])
        node={}
        node[optinum_attr]={}
        attr=[i for i in attr if i!= optinum_attr]
        for val in attr_val:
          #extraxting the subdata of optimum atrribute
          attr_data=X.where(X[optinum_attr]==val).dropna()
          target_attr_data=y.where(X[optinum_attr]==val).dropna()
          node[optinum_attr][val]=self.discrete_real(attr_data,target_attr_data,depth+1,attr,parent)
        
        self.tree=node
        return node
    #################################################################################################################################
       
    #Functions for discrete output
    def check_all_same(self,y):
      values,cout = np.unique(y,return_counts=True)   
      return len(cout)
    def max_occuring(self,y):
      return np.bincount(y).argmax() 
      
      
