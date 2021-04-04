# ES654-2020 Assignment 3

*Hetvi Shastri* - *18110064*

------

**Autograd fit**

Learning rate:-0.01

Number of iterations:-100

Learning rate function over time:- Constant

Batch size=30

- Including Bias

    - RMSE:-1.0013
    - MAE:-0.8512

- Without Bias

    - RMSE:-1.0116
    - MAE:-0.8169

Learning rate function over time:- Inverse

- Including Bias

    - RMSE:-2.32
    - MAE:-1.95

- Without Bias

    - RMSE:-2.35
    - MAE:-1.95

Here we can see that there is increase in errors if we change the learning type from constant to inverse as the value of learning rate is less. After updating learning rate to higher number inverse will be better than constant. 

By varying value of batch size if we compare between SGD and mini batch GD. SGD is performing better. But by decreasing batch size there will be increase in time.

By increasing number of iterations there will be increase in accuracy but there will be increase in time.