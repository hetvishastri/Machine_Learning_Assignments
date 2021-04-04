import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
np.random.seed(10)  #Setting seed for reproducibility

degree=[1,3,5,7,9]
N=[]

for i in range(60,100,4):
    N.append(i)

mag_theta_d=[]
for n in N:
    print(n)
    mag_theta=[]
    x = np.array([i*np.pi/180 for i in range(60,300+4*(n-60),4)])
    y = 4*x + 7 + np.random.normal(0,3,len(x))
    for d in degree:
        X=[]
        poly = PolynomialFeatures(d)
        for i in range(len(x)):
            X.append(poly.transform([x[i]]))
        X=pd.DataFrame(X)
        y=pd.Series(y)
        X=X.iloc[:,1:]
        LR = LinearRegression(fit_intercept=True)
        LR.fit_vectorised(X, y,len(X),20,0.05) 
        theta=LR.theta()
        norm=np.linalg.norm(theta) 
        mag_theta.append(norm)
    mag_theta_d.append(mag_theta)

plt.figure()
for i in range(len(N)):
    plt.semilogy(degree,mag_theta_d[i],label='N='+str(N[i])+" No of samples")
plt.legend()
plt.title("Magnitude of theta vs Degree with varying samples")
plt.xlabel("Degree")
plt.ylabel("log(|theta|")
plt.savefig("6.png")


