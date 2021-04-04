### Questions

1. Complete the decision tree implementation in tree/base.py. **[5 marks]**
The code should be written in Python and not use existing libraries other than the ones already imported in the code. Your decision tree should work for four cases: i) discrete features, discrete output; ii) discrete features, real output; iii) real features, discrete output; real features, real output. Your decision tree should be able to use GiniIndex or InformationGain as the criteria for splitting. Your code should also be able to plot/display the decision tree. 

    > You should be editing the following files.
  
    - `metrics.py`: Complete the performance metrics functions in this file. 

    - `usage.py`: Run this file to check your solutions.

    - tree (Directory): Module for decision tree.
      - `base.py` : Complete Decision Tree Class.
      - `utils.py`: Complete all utility functions.
      - `__init__.py`: **Do not edit this**

    > You should run _usage.py_ to check your solutions. 

2. 
    a) Show the usage of *your decision tree* on the [IRIS](https://archive.ics.uci.edu/ml/datasets/Iris) dataset. The first 70% of the data should be used for training purposes and the remaining 30% for test purposes. Show the accuracy, per-class precision and recall of the decision tree you implemented on the test dataset. **[1 mark]**

    b) Use 5 fold cross-validation on the dataset. Using nested cross-validation find the optimum depth of the tree. **[2 marks]**
    
    > You should be editing `iris-experiments.py` for the code containing the experiments.

3. 
    a) Show the usage of your decision tree for the [real estate price prediction regression](https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set) problem. **[1 mark]**
    
    b) Compare the performance of your model with the decision tree module from scikit learn. **[1 mark]**
    
   > You should be editing `estate-experiments.py` for the code containing the experiments.
    
4. Create some fake data to do some experiments on the runtime complexity of your decision tree algorithm. Create a dataset with N samples and M binary features. Vary M and N to plot the time taken for: 1) learning the tree, 2) predicting for test data. How do these results compare with theoretical time complexity for decision tree creation and prediction. You should do the comparison for all the four cases of decision trees. **[2 marks]**	

    >You should be editing `experiments.py` for the code containing the experiments. 

5. 
    a) Implement Adaboost on Decision Stump (depth -1 tree). You could use Decision Tree learnt in assignment #1 or sklearn decision tree and solve it for the case of real input and discrete output. Edit `ensemble/ADABoost.py` **[2 marks]**

    b) Implement AdaBoostClassifier on Iris data set. Fix a random seed of 42. Shuffle the dataset according to this random seed. Use the first 60% of the data for training and last 40% of the data set for testing. Using sepal width and petal width as the two features, plot the decision surfaces as done for Q1a) and compare the accuracy of AdaBoostClassifier using 3 estimators over decision stump. Include your code in `q5_ADABoost.py`. [*We will be solving the problem in 2 class setting. The two classes are: virginica and not virginica.  **[2 marks]**

6.
    a) Implement Bagging(BaseModel, num_estimators): where base model is be DecisionTree (or sklearn decision tree) you have implemented. In a later assignment, you would have to implement the above over LinearRegression() also. Edit `ensemble/bagging.py`. Use `q6_Bagging.py` for testing.[*We will be implementing only for DecisionTrees [2 marks*]]

    
7. 
    a) Implement RandomForestClassifier() and RandomForestRegressor() classes in `tree/randomForest.py`. Use `q7_RandomForest.py` for testing.[*2 marks*]

     b) Generate the plots for Iris data set. Fix a random seed of 42. Shuffle the dataset according to this random seed. Use the first 60% of the data for training and last 40% of the data set for testing. Using sepal width and petal width as the two features. Include you code in `random_forest_iris.py`[*2 marks*]


You can answer the subjectve questions (timing analysis, displaying plots) by creating `assignment_q<question-number>_subjective_answers.md`

Doubts about the assignment may be clarified here: https://iitgnacin-my.sharepoint.com/:w:/g/personal/nipun_batra_iitgn_ac_in/EZOsxJwGPFFLhJ-9XmPL0IEBEvpkVz935Bd-nblaVkEzOQ?e=MuXphv
