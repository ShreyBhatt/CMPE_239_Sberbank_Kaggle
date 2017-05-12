import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import model_selection, preprocessing

"""XGBoost would be a good regression model to be used because - it is extremely fast,
it can handle the missing values in our dataset and also it has built-in cross validation.
So these are the things that are extremely important which is handled by this library."""

macro_features = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
"micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
"income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build", "timestamp"]

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
macro = pd.read_csv("macro.csv", usecols=macro_features)

macro_train = pd.merge_ordered(train, macro, on='timestamp', how='left')
macro_test = pd.merge_ordered(test, macro, on='timestamp', how='left')

id_test = macro_test.id
y_train = macro_train["price_doc"]
x_train = macro_train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = macro_test.drop(["id", "timestamp"], axis=1)

print x_train.shape

#Step1 - Perform preprocessing
for c in x_train.columns:
    if x_train[c].dtype == 'object':
        label = preprocessing.LabelEncoder()
        label.fit(list(x_train[c].values))
        x_train[c] = label.transform(list(x_train[c].values))

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        label = preprocessing.LabelEncoder()
        label.fit(list(x_test[c].values))
        x_test[c] = label.transform(list(x_test[c].values))

"""
eta - Makes the model more robust by shrinking the weights on each step .
max_depth - The maximum depth of a tree.
subsample - Denotes the fraction of observations to be randomly samples for each tree.
colsample_bytree - Denotes the fraction of columns to be randomly samples for each tree.
"""
#Step2 -  Build Model
xgb_params = {
    'eta': 0.02,
    'max_depth': 9,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

trainMatrix = xgb.DMatrix(x_train, y_train)
testMatrix = xgb.DMatrix(x_test)

output = xgb.cv(xgb_params, trainMatrix, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)
num_boost_rounds = len(output)

#Step3 - Train the model
model = xgb.train(dict(xgb_params, silent=0), trainMatrix, num_boost_round= num_boost_rounds)

#Step4 - Predict
predictions = model.predict(testMatrix)

#Step5 - Print results
output = pd.DataFrame({'id': id_test, 'price_doc': predictions})
output.head()
output.to_csv('result_with_macro.csv', index=False)


#RESULT - 0.3240(RMSE score)
