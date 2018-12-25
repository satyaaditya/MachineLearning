"""
https://www.kaggle.com/sagarnildass/predicting-boston-house-prices/notebook
https://machinelearningmastery.com/evaluate-performance-machine-learning-algorithms-python-using-resampling/
https://stackoverflow.com/questions/37367405/python-scikit-lear-cant-handle-mix-of-multiclass-and-continuous
https://stackoverflow.com/questions/43613443/difference-between-cross-val-score-and-cross-val-predict
https://stackoverflow.com/questions/41458834/how-is-scikit-learn-cross-val-predict-accuracy-score-calculated/41524968#41524968
"""
"""
Feature description
RM - no :of rooms in home
LSTAT : lower class neighbourhood
PTRATIO : teacher to student ratio
"""
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn import linear_model
from sklearn import metrics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    """returns prices and features"""
    data = pd.read_csv('datasets/housing.csv')
    prices = data['MEDV']
    features = data.drop(['MEDV'], axis=1)    # remove it from data as we need to predict it
    print(data.head())  #   prints top columns 5 for ex
    return [features, prices]

def calculate_descriptive_statstics(prices):
    minumum_price = np.min(prices)
    maximum_price = np.max(prices)
    mean_price = np.mean(prices)
    median_price = np.median(prices)
    standard_deviation = np.std(prices)
    print ('minumum_price' , minumum_price)
    print ('maximum_price', maximum_price)
    print('mean price', mean_price)
    print('median price', median_price)
    print('standard_deviation', standard_deviation)


def do_predictions_test_split(features, prices):
    X_train, X_test, y_train, y_test = train_test_split(features, prices)
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(X_train, y_train)
    predictions = linear_regression.predict(X_test)
    print('---------------------------------------------')
    print('accuracy -', linear_regression.score(X_test, y_test))
    print('variance R2 through train_test_split', performance_metric(y_test, predictions))
    pass

def do_prediction_k_fold(features, prices):
    linear_regression_model = linear_model.LinearRegression()
    kfold = KFold(n_splits=10, random_state=7)
    result = cross_val_score(linear_regression_model, features, prices, cv=kfold)
    print('---------------------------------------------')
    # print('accuracy -', result.mean() * 100.0)
    print('standard deviation', result.std()*100.0)
    print('---------------------------------------------')
    predicted = cross_val_predict(linear_regression_model, features, prices, cv=10)
    print('variance R2 through kfold', performance_metric(prices, predicted))

def linear_regression():
    features, prices = load_data()
    calculate_descriptive_statstics(prices)
    # visualize(features, prices)
    do_predictions_test_split(features, prices)
    do_prediction_k_fold(features, prices)

def visualize(features, prices):
    columns = features.columns
    plt.figure(figsize=(20, 5))
    for i, col in enumerate(columns):
        plt.subplot(1, 3, i + 1)
        x = features[col]
        y = prices
        plt.plot(x, y, 'o')
        # Create regression line
        plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('prices')
    plt.show()


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)

    # Return the score
    return score

if __name__ == "__main__":
    linear_regression()
