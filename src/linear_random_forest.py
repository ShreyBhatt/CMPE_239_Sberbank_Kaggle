import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import make_pipeline


def rmsle_score(y_true_log, y_pred_log):
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    return np.sqrt(np.mean(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))


def score_model(model, pipe, x_train, y_train, x_test, y_test):
    train_error = rmsle_score(y_train, model.predict(pipe.transform(x_train)))
    test_error = rmsle_score(y_test, model.predict(pipe.transform(x_test)))
    return test_error
    # return train_error, test_error


def load_data(with_macro=False):
    train = pd.read_csv("train.csv", parse_dates=['timestamp'])
    macro = pd.read_csv("macro.csv", parse_dates=['timestamp'])
    test = pd.read_csv("test.csv", parse_dates=['timestamp'])
    if with_macro:
        macro_train = pd.merge_ordered(train, macro, on='timestamp', how='left')
        macro_test = pd.merge_ordered(test, macro, on='timestamp', how='left')
        return macro_train, macro, macro_test
    else:
        return train, macro, test


def preprocess_data(train, test):

    # Take log of price for better distribution. Drop id, timestamp and price columns
    y_train = np.log(train["price_doc"])
    x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
    x_test = test.drop(["id", "timestamp"], axis=1)

    # Convert categorical values to dummy columns
    x_train = pd.get_dummies(x_train).astype(np.float64)
    x_test = pd.get_dummies(x_test).astype(np.float64)
    x_train = x_train.drop(['sub_area_Poselenie Klenovskoe'], axis=1)

    # Make a pipeline that transforms X
    # Imputer takes care of missing values and standard scaler normalizes data
    pipe = make_pipeline(Imputer(), StandardScaler())
    pipe.fit(x_train)
    # pipe.transform(X_train)
    return x_train, y_train, x_test, pipe


def regression(x_train, y_train, pipe):

    # Generate training and test data in 80/20 split
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2)
    # print len(X_train), len(X_test), len(Y_train), len(Y_test)

    # Linear Regression, sometimes predicts very small value which makes it infinite when its exponent is taken.
    lr = LinearRegression(fit_intercept=True)
    lr.fit(pipe.transform(X_train), Y_train)
    linear_score = (score_model(lr, pipe, X_train, Y_train, X_test, Y_test))
    # print("Train error: {:.4f}, Test error: {:.4f}".format(*score_model(lr, pipe)))

    # Random Forest Regression, used 1, 10, 50 and 100 number of trees to see which works best.
    rfr = RandomForestRegressor(n_estimators=50, min_samples_leaf=50, n_jobs=-1)
    rfr.fit(pipe.transform(X_train), Y_train)
    forest_score = (score_model(rfr, pipe, X_train, Y_train, X_test, Y_test))
    # print("Train error: {:.4f}, Test error: {:.4f}".format(*score_model(rfr, pipe)))

    return linear_score, forest_score


def start(with_macro=False):
    print "Loading data, with macro: " + str(with_macro)
    train, macro, test = load_data(with_macro)
    linear_score = []
    forest_score = []
    for i in range(0, 5):
        try:
            print "Iteraton: "+str(i)
            x_train, y_train, x_test, pipe = preprocess_data(train, test)
            score1, score2 = regression(x_train, y_train, pipe)
            print "Linear Regression Score: " + str(score1)
            print "Random Forest Regression Score: " + str(score2)

            if score1 != float('inf'):
                linear_score.append(score1)
            else:
                linear_score.append(1.0)

            forest_score.append(score2)

        except Exception:
            print "exception"
            continue
    print ""
    print "Mean Scores after "+str(i)+" iterations:"
    print "Linear Regression Score: " + str(np.mean(linear_score))
    print "Random Forest Regression Score: " + str(np.mean(forest_score))


if __name__ == '__main__':
    w_macro = False
    start(w_macro)
