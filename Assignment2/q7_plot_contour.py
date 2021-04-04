import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y=4*X.iloc[:,0]+4
y=y.values.reshape(len(y),)
y = pd.Series(y)


LR = LinearRegression(fit_intercept=True)
LR.fit_vectorised(X, y,n_iter=200,lr=0.01) 
y_hat = LR.predict(X)
LR.plot_surface(X,y,1,1)

