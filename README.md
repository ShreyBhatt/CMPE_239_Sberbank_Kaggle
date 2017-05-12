# CMPE_239_Sberbank_Kaggle
CMPE 239 Project: SberBank Housing Market Challenge Kaggle

Dataset Explanation:

File Name and Size: train.csv (45 MB), test.csv (11 MB), macro.csv (1.5 MB)
Dimensions of data: train.csv(30471*292), test.csv(7662*291), macro.csv(2484*100)

incase XGBoost is not installed, install it using:

$ pip install xgboost

execute files in command line using following command

$ python file_name.py 


List of files:

linear_random_forest.py : consists code for Linear Regression and Logistic Regression with and without using macros. 
			  File might take a bit longer to run as it uses random forests and the data set is quite large.

xgbMacro.py 		: consists code for XGBoost Regression with using macro

xgbwoMacro.py 		: consists code for XGBoost Regression without using macro

macroLSTM.py 		: consists code for lstm with macro

Data_Exploration.ipynb is the jupyter notebook which has all the data exploration and graph
run jupyter notebook in terminal in the same directory and run all the cells on opening the file.

Reference:

1)For lstm.py
https://keras.io/layers/recurrent/ 
https://github.com/fchollet/keras

2)For XGBoost
https://github.com/dmlc/xgboost




