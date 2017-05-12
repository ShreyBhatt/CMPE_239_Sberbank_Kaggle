import numpy as np
import pandas as pd
from keras.layers import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time
from sklearn import model_selection, preprocessing

"""Experimental Doesn't work""""

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
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values))
        x_train[c] = lbl.transform(list(x_train[c].values))

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values))
        x_test[c] = lbl.transform(list(x_test[c].values))


#Step2 -  Build Model
model = Sequential()
model.add(LSTM(
    input_dim=1,
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')

model.fit(
    x_train,
    y_train,
    batch_size=512,
    nb_epoch=1,
    validation_split=0.05)

#Step3 - Predictions
predictions = lstm.predict_sequences_multiple(model, x_test, 50, 50)
