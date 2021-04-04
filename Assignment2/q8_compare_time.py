import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import time

np.random.seed(42)


N = 15
time_grad=[]
time_norm=[]
#Varying number of features

for i in range(10,3000,10):
    P = i
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    
    LR = LinearRegression(fit_intercept=True)
    start_time=time.time()
    LR.fit_vectorised(X, y,n_iter=100) 
    end_time=time.time()
    time_grad.append(end_time-start_time)
    start_time=time.time()
    LR.fit_normal(X, y) 
    end_time=time.time()
    time_norm.append(end_time-start_time)
    
    print(i)


num_features=[j for j in range(10,3000,10)]
plt.figure()
plt.plot(num_features,time_grad)
plt.plot(num_features,time_norm)
plt.legend(["Gradient descent","Normal"])
plt.xlabel('Number of features')
plt.ylabel("Fit time")
plt.title('Fit time Vs number of features')
plt.savefig("1.png")


#Varying number of samples
P = 15
time_grad=[]
time_norm=[]

for i in range(10,1000,10):
    N = i
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))
    
    LR = LinearRegression(fit_intercept=True)
    start_time=time.time()
    LR.fit_vectorised(X, y,n_iter=100) 
    end_time=time.time()
    time_grad.append(end_time-start_time)

    start_time=time.time()
    LR.fit_normal(X, y) 
    end_time=time.time()
    time_norm.append(end_time-start_time)
    print(i)


num_features=[j for j in range(10,1000,10)]
plt.figure()
plt.plot(num_features,time_grad)
plt.plot(num_features,time_norm)
plt.legend(["Gradient descent","Normal"])
plt.xlabel('Number of samples')
plt.ylabel("Fit time")
plt.title('Fit time Vs number of samples')
plt.savefig("2.png")


#Varying number of iterations
P = 15
N=15
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))
time_grad=[]
time_norm=[]

for i in range(1,1000):
    LR = LinearRegression(fit_intercept=True)
    start_time=time.time()
    LR.fit_vectorised(X, y,n_iter=i) 
    end_time=time.time()
    time_grad.append(end_time-start_time)

    start_time=time.time()
    LR.fit_normal(X, y) 
    end_time=time.time()
    time_norm.append(end_time-start_time)
    print(i)


num_features=[j for j in range(1,1000)]
plt.figure()
plt.plot(num_features,time_grad)
plt.plot(num_features,time_norm)
plt.legend(["Gradient descent","Normal"])
plt.xlabel('Number of iterations')
plt.ylabel("Fit time")
plt.title('Fit time Vs number of iterations')
plt.savefig("3.png")





