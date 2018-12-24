#importing linear regression
from sklearn import linear_model

from sklearn import datasets
import numpy as np
import pandas as pd

def linear_regression():
    # ---------------------- load data ------------------------------------------------
    #load boston data set
    data = datasets.load_boston()

    #take dataset and load it in padas dataframe and setting independent variables
    independent_data_frame = pd.DataFrame(data.data[:-10], columns=data.feature_names)

    # print list(independent_data_frame.columns.values)
    print data.feature_names

    test_data_frame = pd.DataFrame(data.data[-10:], columns=data.feature_names)

    #load dependent or target variables in another data frame
    dependent_data_frame = pd.DataFrame(data.target[:-10], columns=['MEDV'])

    print list(dependent_data_frame.columns.values)

    # y = mX + c
    X = independent_data_frame
    y = dependent_data_frame['MEDV']

    # ----------------------- fit data in to model ------------------------------------------------

    # instantiate Linear regression class
    linear_regression = linear_model.LinearRegression()

    #fit data in to linear model
    linear_regression.fit(X, y)


    # ------------------------ make predicitions --------------------------------------------------
    predictions = linear_regression.predict(test_data_frame)
    print predictions[:5]
    # R sqr
    print linear_regression.score(X, y)
    print linear_regression.coef_

if __name__ == "__main__":
    linear_regression()