import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing.polynomial_features import PolynomialFeatures

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

degree=[]
for i in range(10):
    degree.append(i)


def fit_normal(X, y):
    XT=X.T
    X=np.array(X)
    XT=np.array(XT)
    y_1=np.array(y)
    theta=((np.linalg.inv(XT.dot(X))).dot(XT)).dot(y_1)
    return theta

#varying degree
mag_theta=[]
for d in degree:
    X=[]
    #applying polynomial transform on every sample to create a dataframe
    for i in range(len(x)):
        poly = PolynomialFeatures(d)
        X.append(poly.transform([x[i]]))
    X=pd.DataFrame(X)
    y=pd.Series(y)
    theta=fit_normal(X,y)
    mag_theta.append(np.linalg.norm(theta))

plt.figure()
plt.plot(degree,mag_theta)
plt.title("Theta v/s degree")
plt.xlabel("degree")
plt.ylabel("|theta|")
plt.savefig("5.png")

           
        



