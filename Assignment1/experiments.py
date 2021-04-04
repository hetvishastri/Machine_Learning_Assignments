
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
def calculate(X,y):
    tree=DecisionTree("information_gain")
    start_fit=time.time()
    tree.fit(X,y)
    end_fit=time.time()
    time_fit=end_fit-start_fit
    
    start_predict=time.time()
    tree.predict(X)
    end_predict=time.time()
    time_predict=end_predict-start_predict

    return time_fit,time_predict

# ...
# Function to plot the results


# ..
# Function to create fake data (take inspiration from usage.py)
# ...
def create_fake_data(N,M,flag):
    if(flag==0):
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
    elif(flag==1):
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(M, size = N), dtype="category")
    elif(flag==2):
        X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randint(M, size = N),  dtype="category")
    else:
        X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randn(N))
    return X,y
#
M=5 #Features
Num_sample=[i for i in range(1,50)]
time_f1=[]
time_P1=[]
time_f2=[]
time_P2=[]
time_f3=[]
time_P3=[]
time_f4=[]
time_P4=[]

for N in Num_sample:
    #Case 1 Real input Real output
    flag=0
    X,y=create_fake_data(N,M,flag)
    time_fit1,time_predict1=calculate(X,y)
    time_f1.append(time_fit1)
    time_P1.append(time_predict1)
    #Case 2 Real input dicrete output
    flag=1
    X,y=create_fake_data(N,M,flag)
    time_fit2,time_predict2=calculate(X,y)
    
    time_f2.append(time_fit2)
    time_P2.append(time_predict2)
    #Case 3 Discrete input discrete output
    flag=2
    X,y=create_fake_data(N,M,flag)
    time_fit3,time_predict3=calculate(X,y)

    time_f3.append(time_fit3)
    time_P3.append(time_predict3)
    #Case 4 Discrete input real output
    flag=3
    X,y=create_fake_data(N,M,flag)
    time_fit4,time_predict4=calculate(X,y)
    time_f4.append(time_fit4)
    time_P4.append(time_predict4)

plt.figure()
plt.plot(Num_sample,time_f1)
plt.xlabel('Number of samples')
plt.ylabel("Fit time")
plt.title('Fit time Vs number of sample(RIRO) M=5')
plt.savefig("RIRO_N_fit.png")

plt.figure()
plt.plot(Num_sample,time_P1)
plt.xlabel('Number of samples')
plt.ylabel("Predict time")
plt.title('Predict time Vs number of sample(RIRO) M=5')
plt.savefig("RIRO_N_predict.png")

plt.figure()
plt.plot(Num_sample,time_f2)
plt.xlabel('Number of samples')
plt.ylabel("Fit time")
plt.title('Fit time Vs number of sample(RIDO) M=5')
plt.savefig("RIDO_N_fit.png")

plt.figure()
plt.plot(Num_sample,time_P2)
plt.xlabel('Number of samples')
plt.ylabel("Predict time")
plt.title('Predict time Vs number of sample(RIDO) M=5')
plt.savefig("RIDO_N_predict.png")

plt.figure()
plt.plot(Num_sample,time_f3)
plt.xlabel('Number of samples')
plt.ylabel("Fit time")
plt.title('Fit time Vs number of sample(DIDO) M=5')
plt.savefig("DIDO_N_fit.png")

plt.figure()
plt.plot(Num_sample,time_P3)
plt.xlabel('Number of samples')
plt.ylabel("Predict time")
plt.title('Predict time Vs number of sample(DIDO) M=5')
plt.savefig("DIDO_N_predict.png")

plt.figure()
plt.plot(Num_sample,time_f4)
plt.xlabel('Number of samples')
plt.ylabel("Fit time")
plt.title('Fit time Vs number of sample(DIRO) M=5')
plt.savefig("DIRO_N_fit.png")

plt.figure()
plt.plot(Num_sample,time_P4)
plt.xlabel('Number of samples')
plt.ylabel("Predict time")
plt.title('Predict time Vs number of sample(DIRO) M=5')
plt.savefig("DIRO_N_predict.png")


Num_sample=30 #Features
Num_features=[i for i in range(1,50)]
time_f1=[]
time_P1=[]
time_f2=[]
time_P2=[]
time_f3=[]
time_P3=[]
time_f4=[]
time_P4=[]

for M in Num_features:
    print(M)
    #Case 1 Real input Real output
    flag=0
    X,y=create_fake_data(N,M,flag)
    time_fit1,time_predict1=calculate(X,y)
    time_f1.append(time_fit1)
    time_P1.append(time_predict1)
    #Case 2 Real input dicrete output
    flag=1
    X,y=create_fake_data(N,M,flag)
    time_fit2,time_predict3=calculate(X,y)
    time_f2.append(time_fit2)
    time_P2.append(time_predict2)
    #Case 3 Discrete input discrete output
    flag=2
    X,y=create_fake_data(N,M,flag)
    time_fit3,time_predict3=calculate(X,y)
    time_f3.append(time_fit3)
    time_P3.append(time_predict3)
    #Case 4 Discrete input real output
    flag=3
    X,y=create_fake_data(N,M,flag)
    time_fit4,predict4=calculate(X,y)
    time_f4.append(time_fit4)
    time_P4.append(time_predict4)

plt.figure()
plt.plot(np.array(Num_features),time_f1)
plt.xlabel('Number of samples')
plt.ylabel("Fit time")
plt.title('Fit time Vs number of sample(RIRO) N=30')
plt.savefig("RIRO_M_fit.png")

plt.figure()
plt.plot(Num_features,time_P1)
plt.xlabel('Number of samples')
plt.ylabel("Predict time")
plt.title('Predict time Vs number of sample(RIRO) N=30')
plt.savefig("RIRO_M_predict.png")

plt.figure()
plt.plot(Num_features,time_f2)
plt.xlabel('Number of samples')
plt.ylabel("Fit time")
plt.title('Fit time Vs number of sample(RIDO) N=30')
plt.savefig("RIDO_M_fit.png")

plt.figure()
plt.plot(Num_features,time_P2)
plt.xlabel('Number of samples')
plt.ylabel("Predict time")
plt.title('Predict time Vs number of sample(RIDO) N=30')
plt.savefig("RIDO_M_predict.png")

plt.figure()
plt.plot(Num_features,time_f3)
plt.xlabel('Number of samples')
plt.ylabel("Fit time")
plt.title('Fit time Vs number of sample(DIDO) N=30')
plt.savefig("DIDO_M_fir.png")

plt.figure()
plt.plot(Num_features,time_P3)
plt.xlabel('Number of samples')
plt.ylabel("Predict time")
plt.title('Predict time Vs number of sample(DIDO) N=30')
plt.savefig("DIDO_M_predict.png")

plt.figure()
plt.plot(Num_features,time_f4)
plt.xlabel('Number of samples')
plt.ylabel("Fit time")
plt.title('Fit time Vs number of sample(DIRO) N=30')
plt.savefig("DIRO_M_fit.png")

plt.figure()
plt.plot(Num_features,time_P4)
plt.xlabel('Number of samples')
plt.ylabel("Predict time")
plt.title('Predict time Vs number of sample(DIRO) N=30')
plt.savefig("DIRO_M_predict.png")
# ..other functions
