
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    count=0.0
    for i in range(len(y)):
      if(y_hat[i]==y[i]):
        count=count+1
    return count*100/len(y)

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size == y.size)
    y_correct=y.where(y==cls).dropna()
    pred=0.0
    y_hat_pred_correct=y_hat.where(y==cls).dropna()
    if(len(y_correct)!=0):
        pred=len(y_hat_pred_correct)/len(y_correct)
    return pred

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size == y.size)
    y_correct=y.where(y_hat==cls).dropna()
    y_hat_pred_correct=y_hat.where(y_hat==cls).dropna()
    if(len(y_correct)==0):
      return 0.0
    recall=len(y_hat_pred_correct)/len(y_correct)
    return recall

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    assert(y_hat.size == y.size)
    error=y_hat-y
    rmse=((error**2).mean())**0.5
    return rmse

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    assert(y_hat.size == y.size)
    abs_error=abs(y_hat-y)
    mae=abs_error.mean()
    return mae

