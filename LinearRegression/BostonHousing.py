#importing linear regression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot  as plt

from sklearn import datasets
import numpy as np
import pandas as pd

def linear_regression():
    # ---------------------- load data ------------------------------------------------
    #load boston data set
    data = datasets.load_boston()

    #take dataset and load it in padas dataframe and setting independent variables
    independent_data_frame = pd.DataFrame(data.data, columns=data.feature_names )#data.feature_names)

    # print list(independent_data_frame.columns.values)
    # print data.feature_names

    #load dependent or target variables in another data frame
    dependent_data_frame = pd.DataFrame(data.target, columns=['MEDV'])

    # print list(dependent_data_frame.columns.values)

    # y = mX + c
    X = independent_data_frame
    y = dependent_data_frame['MEDV']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print X_train.shape, y_train.shape
    print X_test.shape, y_test.shape
    # ----------------------- fit data in to model ------------------------------------------------

    # instantiate Linear regression class
    linear_regression = linear_model.LinearRegression()

    #fit data in to linear model
    linear_regression.fit(X_train, y_train)


    # ------------------------ make predicitions --------------------------------------------------
    predictions = linear_regression.predict(X_test)
    print predictions[:10]


    plt.scatter(y_test, predictions, edgecolors='red')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()
    # R sqr
    print linear_regression.score(X, y)
    print linear_regression.coef_

if __name__ == "__main__":
    linear_regression()
