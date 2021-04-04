
## Vary M and N to plot the time taken for: 
1) learning the tree

    - **Real input Real output**

        - Iterating over number of samples
![img](RIRO_N_fit.png)

        - Iterating over number of features
![img](RIRO_M_fit.png)

            

    - **Real input discrete output**

        - Iterating over number of samples          
![img](RIDO_N_fit.png)

        - Iterating over number of features     
![img](RIDO_M_fit.png)

    
    - **Discrete input discrete output**

        - Iterating over number of samples
![img](DIDO_N_fit.png)

        - Iterating over number of features
![img](DIDO_M_fir.png)

    - **Discrete input real output**

        - Iterating over number of samples
![img](DIRO_N_fit.png)

        - Iterating over number of features
![img](DIRO_M_fit.png)

1) learning the tree

    - **Real input Real output**

        - Iterating over number of samples
![img](RIRO_N_predict.png)

        - Iterating over number of features
![img](RIRO_M_predict.png)

            

    - **Real input discrete output**

        - Iterating over number of samples
![img](RIDO_N_predict.png)

        - Iterating over number of features
![img](RIDO_M_predict.png)

    
    - **Discrete input discrete output**

        - Iterating over number of samples
![img](DIDO_N_predict.png)

        - Iterating over number of features
![img](DIDO_M_predict.png)

    - **Discrete input real output**

        - Iterating over number of samples
![img](DIRO_N_predict.png)

        - Iterating over number of features
![img](DIRO_M_predict.png)

## How do these results compare with theoretical time complexity for decision tree creation and prediction. You should do the comparison for all the four cases of decision trees. 

Theoretically time coplexity for fitting is O(mnlog(n)) where m is number of features and n is number of samples. So as value of m and n increases there will be increase in time. This is visible through above plots.Some curve is visible due to log(n) part.As we are finding best splits hence there may be peaks in between.

Theoretically time coplexity for predicting is O(log(n)) n is number of samples. This must be between linearly increasing and constant time complexity. As we are finding best splits hence there may be peaks in between.



