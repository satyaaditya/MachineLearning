"""
references -
    dataset - https://www.kaggle.com/rajanchavda1/price-prediction/data
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


def load_data(file_loc):
    data = pd.read_csv(file_loc)
    target = data.Present_Price
    features = data.drop(['Car_Name', 'Present_Price', 'Seller_Type', 'Owner', 'Transmission'], axis=1)
    # print(features.head())
    features['Fuel_Type'] = features.Fuel_Type.apply(lambda x: 1 if x == 'Petrol' else 0)
    return [features, target]

def do_predictions_test_split(features, target):
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(features, target)
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
    y_predictions = model.predict(X_test)
    print('Mean Squared Error: ', mean_squared_error(y_test, y_predictions), '\n\n\n')
    print('accuracy - ', performance_metric(y_test, y_predictions))

    features, target = load_data('datasets/test.csv')
    y_predictions = model.predict(features)
    print(target)
    print(y_predictions)
    print('Mean Squared Error: ', mean_squared_error(target, y_predictions), '\n\n\n')

    pass

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    # Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)

    # Return the score
    return score

def linear_regression():
    features, target = load_data('datasets/car_data.csv')

    do_predictions_test_split(features, target)



if __name__ == "__main__":
    linear_regression()
