#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rc('figure', max_open_warning = 0)
# Import Autograd modules here
from autograd import elementwise_grad
import autograd.numpy as np
import imageio
import os

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods

    def fit_non_vectorised(self, X, y, batch_size=30, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        self.n_iter=n_iter
        self.theta_iter0=[]
        self.theta_iter1=[]
        self.error=[]
        
        num_sample=np.shape(y)[0]
        features=np.shape(X)[1]
        if(self.fit_intercept==True):
            theta=np.random.randn(features+1,)
            X=np.array(X)
            column=np.ones((num_sample,1))
            X=np.append(column,X,axis=1)
            X=pd.DataFrame(X)
        else:
            theta=np.random.randn(features,)
        for i in range(n_iter):
            batches=self.create_batch(X,y,batch_size)
            if(lr_type=='constant'):
                alpha=lr
            else:
                alpha=lr/(i+1)
            for batch in batches:
                [batch_X,batch_y]=batch
                for j in range(np.shape(theta)[0]):
                    y_hat=batch_X.dot(theta)
                    #theta0=theta0-alpha*(2*(X.T))((X)theta0 - y)/n
                    theta[j]=theta[j]-(2/batch_size)*alpha*(batch_X.iloc[:,j].T.dot(y_hat-batch_y))
            #storing for further plotting
            self.theta_iter0.append(theta[0])
            self.theta_iter1.append(theta[1])
            self.error.append(self.rss(X.iloc[:,0:2],y,theta[0:2]))
        self.coef_=theta 

        

    def fit_vectorised(self, X, y,batch_size=30, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        self.n_iter=n_iter
        self.theta_iter0=[]
        self.theta_iter1=[]
        self.error=[]
        
        num_sample=np.shape(y)[0]
        features=np.shape(X)[1]
        if(self.fit_intercept==True):
            theta=np.random.randn(features+1,)
            X=np.array(X)
            column=np.ones((num_sample,1))
            X=np.append(column,X,axis=1)
            X=pd.DataFrame(X)
            
        else:
            theta=np.random.randn(features,)
        for i in range(n_iter):
            batches=self.create_batch(X,y,batch_size)
            if(lr_type=='constant'):
                alpha=lr
            else:
                alpha=lr/(i+1)
            for batch in batches:
                [batch_X,batch_y]=batch
                y_hat=batch_X.dot(theta)
                theta=theta-(2/batch_size)*alpha*(batch_X.T.dot(y_hat-batch_y))
            self.theta_iter0.append(theta[0])
            self.theta_iter1.append(theta[1])
            self.error.append(self.rss(X.iloc[:,0:2],y,theta[0:2]))
        self.coef_=theta 
        
        

    def fit_autograd(self, X, y, batch_size=30, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        self.n_iter=n_iter
        self.theta_iter0=[]
        self.theta_iter1=[]
        self.error=[]
    
        num_sample=np.shape(y)[0]
        features=np.shape(X)[1]
        if(self.fit_intercept==True):
            theta=np.random.randn(features+1,)
            X=np.array(X)
            column=np.ones((num_sample,1))
            X=np.append(column,X,axis=1)
            X=pd.DataFrame(X)
        else:
            theta=np.random.randn(features,)    
        gradient = elementwise_grad(self.cost_function)
        for i in range(n_iter):
            batches=self.create_batch(X,y,batch_size)
            if(lr_type=='constant'):
                alpha=lr
            else:
                alpha=lr/(i+1)
            for batch in batches:
                [batch_X,batch_y]=batch
                self.batch_X=np.array(batch_X)
                self.batch_y=np.array(batch_y)
                grad_des=gradient(theta)
                theta=theta-alpha*grad_des
            #storing for plotting
            self.theta_iter0.append(theta[0])
            self.theta_iter1.append(theta[1])
            self.error.append(self.rss(X.iloc[:,0:2],y,theta[0:2]))
        self.coef_ = theta
        

        
        

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''

        
        num_sample=np.shape(y)[0]
        if(self.fit_intercept==True):
            X=np.array(X)
            column=np.ones((num_sample,1))
            X=np.append(column,X,axis=1)
            X=pd.DataFrame(X)
        XT=X.T
        X=np.array(X)
        XT=np.array(XT)
        y_1=np.array(y)
        self.coef_=((np.linalg.inv(XT.dot(X))).dot(XT)).dot(y_1)
 

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        row=(np.shape(X)[0])
        X_1=X.copy()
        if(self.fit_intercept==True):
            column=pd.Series(np.ones(row))
            X_1.insert(0,'0',column,allow_duplicates=True)
        y_hat=np.dot(X_1,self.coef_)
        y_hat=y_hat.reshape(row,)
        return y_hat


    def plot_surface(self, X, y, t_0, t_1):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """
        if not os.path.exists("surface"):
            os.mkdir("surface")
        #creating dataset according to fit intercept
        X_1=X.copy()
        num_sample=np.shape(y)[0]
        if(self.fit_intercept==True):
            X_1=np.array(X_1)
            column=np.ones((num_sample,1))
            X_1=np.append(column,X_1,axis=1)
            X_1=pd.DataFrame(X_1)
        #creating meshgrid
        T1,T2=np.meshgrid(np.linspace(0,8,100),np.linspace(-3,8,100))
        error = np.array([self.rss(X_1.iloc[:,0:2],y.values.reshape(-1,1),np.array([t1,t2]).reshape(-1,1)) for t1, t2 in zip(np.ravel(T1), np.ravel(T2)) ] )
        error=error.reshape(T1.shape)


        #iterating across different values of theta and error obtained at every iteration
        filenames=[]
        for i in range(0,self.n_iter,2):
            print(i)
            fig2=plt.figure(figsize=(10,10))
            ax2=Axes3D(fig2)  
            ax2.view_init(10, 30)
            ax2.set_title("error="+"{:.2f}".format(self.error[i]))
            ax2.set_xlabel('c')
            ax2.set_ylabel('m')
            ax2.set_zlabel('error')
            ax2.plot_surface(T1, T2, error, rstride = 5, cstride = 5, cmap = 'jet', alpha=0.5)
            ax2.scatter(self.theta_iter0[i],self.theta_iter1[i],self.error[i],marker='o', s=12**2, color='orange')
            ax2.tick_params(axis='both', which='major', labelsize=15) 
            ax2.plot(self.theta_iter0[0:i],self.theta_iter1[0:i],self.error[0:i],linestyle="dashed",linewidth=2,
             color="grey")   
            filename="surface/"+str(i)+".png"
            filenames.append(filename)
            plt.savefig(filename)

        import imageio
        with imageio.get_writer('surface.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)


        
 


        
    def plot_line_fit(self, X, y, t_0, t_1):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        if not os.path.exists("line"):
            os.mkdir("line")
        X1=X.iloc[:,0]
        X1=np.array(X1)
        X0=np.ones((len(X),))
        filenames=[]
        for i in range(0,self.n_iter,2):
            print(i)
            line=(X0*self.theta_iter0[i]+X1*self.theta_iter1[i])
            plt.figure()
            plt.scatter(X1,y,color='b')
            plt.xlabel("x1")
            plt.ylabel("y")
            
            plt.title("m="+"{:.2f}".format(self.theta_iter1[i])+"  "+"c="+"{:.2f}".format(self.theta_iter1[i]))
            plt.plot(X1,line,'-r')
            filename="line/"+str(i)+".png"
            filenames.append(filename)
            plt.savefig(filename)

        import imageio
        with imageio.get_writer('line.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

    def plot_contour(self, X, y, t_0, t_1):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """
        if not os.path.exists("contour"):
            os.mkdir("contour")
        #creating dataset according to fit intercept
        X_1=X.copy()
        num_sample=np.shape(y)[0]
        if(self.fit_intercept==True):
            X_1=np.array(X_1)
            column=np.ones((num_sample,1))
            X_1=np.append(column,X_1,axis=1)
            X_1=pd.DataFrame(X_1)
        #creating meshgrid
        T1,T2=np.meshgrid(np.linspace(0,8,100),np.linspace(-3,8,100))
        error = np.array([self.rss(X_1.iloc[:,0:2],y.values.reshape(-1,1),np.array([t1,t2]).reshape(-1,1)) for t1, t2 in zip(np.ravel(T1), np.ravel(T2)) ] )
        error=error.reshape(T1.shape)
        #iterating across different values of theta and error obtained at every iteration
        filenames=[]
        for i in range(0,self.n_iter,2):
            print(i)
            fig1,ax1=plt.subplots(1,1)
            ax1.set_xlabel('c')
            ax1.set_ylabel('m')
            ax1.contour(T1, T2, error,100,cmap='jet')
            ax1.scatter(self.theta_iter0[i],self.theta_iter1[i],marker='o', s=12**2, color='orange')
            ax1.tick_params(axis='both', which='major', labelsize=15) 
            ax1.plot(self.theta_iter0[0:i],self.theta_iter1[0:i],linestyle="dashed",linewidth=2,
             color="grey")
            filename="contour/"+str(i)+".png"
            filenames.append(filename)
            plt.savefig(filename)
        import imageio
        with imageio.get_writer('contour.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

    def create_batch(self,X,y,batch_size):
        batches=[]
        data=pd.concat([X,y],axis=1)
        data.sample(frac=1)
        num_of_batches=data.shape[0]//batch_size
        i=0
        #appending the data part wise
        for i in range(num_of_batches):
            batch=data.iloc[i*batch_size:i+1*batch_size,:]
            batch_X=batch.iloc[:,:-1]
            batch_y=batch.iloc[:,-1]
            batches.append([batch_X,batch_y])
        #if data is not covered add the further data
        if data.shape[0]%batch_size!=0:
            batch=data.iloc[i*batch_size:data.shape[0],:]
            batch_X=batch.iloc[:,:-1]
            batch_y=batch.iloc[:,-1]
            batches.append([batch_X,batch_y])
        return batches

    #for autograd
    def cost_function(self,theta):
        pred=np.dot(self.batch_X,theta)
        loss=np.mean((self.batch_y-pred)**2)
        return loss

    def rss(self,X,y,theta): 
        pred=np.dot(X,theta)                 
        e=y-pred
        return np.mean(e**2)

    def theta(self):
        return self.coef_




