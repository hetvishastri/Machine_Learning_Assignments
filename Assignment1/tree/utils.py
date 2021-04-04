
import numpy as np
import pandas as pd
def entropy(Y):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    values,cout=np.unique(Y,return_counts=True)
    entropy=0.0
    
    for i in range(len(values)):
        if(len(Y)>0):
          prob=cout[i]/len(Y)
        if(prob>0):
          entropy+= -prob*np.log2(prob)
    
    return entropy

def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    values,cout=np.unique(Y,return_counts=True)
    gini=0
    
    for i in range(len(values)):
        if(len(Y)>0):
          prob=cout[i]/len(Y)
        if(prob>0):
          gini+= prob*prob
    
    return 1-gini
    
def information_gain(Y, attr):

    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    Entropy=entropy(Y)
    weight_entropy=0
    uniq,num_uniq=np.unique(attr,return_counts=True)

    for i in range(len(uniq)):
        weight=num_uniq[i]/len(Y)
        sub_data=Y.where(attr==uniq[i]).dropna()
        weight_entropy+=weight*entropy(sub_data)
  
    return Entropy-weight_entropy

def information_gain_dr(Y, attr):
    Variance=np.var(Y)
    weight_var=0
    uniq,num_uniq=np.unique(attr,return_counts=True)

    for i in range(len(uniq)):
        weight=num_uniq[i]/len(Y)
        sub_data=Y.where(attr==uniq[i]).dropna()
        weight_var+=weight*np.var(sub_data)

    return Variance-weight_var


def gini_score(Y,attr):
  Gini=gini_index(Y)
  weight_gini=0
  uniq,num_uniq=np.unique(attr,return_counts=True)

  for i in range(len(uniq)):
      weight=num_uniq[i]/len(Y)
      sub_data=Y.where(attr==uniq[i]).dropna()
      weight_gini+=weight*gini_index(sub_data)

  return Gini-weight_gini



    
