from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


def decision_tree():
    # ------------------------------------load the data---------------------------------------
    # names are used for viewing column names in data.headpd.factorize(data['buying'])
    data = pd.read_csv('datasets/car_quality_data.csv',
                       names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

    # print(data.head())  - view top 5 rows in data
    # print(data.info())  - view data information

    # ---------------------------identify target variable and convert it into representable integer value ---------
    data['class'], class_names = pd.factorize(data['class'])
    print('target variable values - ', class_names)
    print('target values after encoding', data['class'].unique())  # after encoding / factorizing

    # ------------------identify independent vairables and convert it into representable integer value ----------
    data['buying'], _ = pd.factorize(data['buying'])
    data['maint'], _ = pd.factorize(data['maint'])
    data['doors'], _ = pd.factorize(data['doors'])
    data['persons'], _ = pd.factorize(data['persons'])
    data['lug_boot'], _ = pd.factorize(data['lug_boot'])
    data['safety'], _ = pd.factorize(data['safety'])
    print('data after factorizing - \n', data.head())
    # print(data.info())


    # -----------------------------------------store dependent and target variables ---------------------------------------
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]  # last column

    # ----------------------------split train and test data -------------------
    '''test_size split data randomly into 70% training and 30% test'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    '''
    CART (Classification and Regression Trees) → uses Gini Index(Classification) as metric.
    ID3 (Iterative Dichotomiser 3) → uses Entropy function and Information gain as metrics.
    '''

    # todo - do with gini  and test for different depths
    # random state - https://stackoverflow.com/questions/39158003/confused-about-random-state-in-decision-tree-of-scikit-learn

    decision_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7, random_state=0)
    decision_tree.fit(X_train, y_train)
    print('accuracy for traning data - ', decision_tree.score(X_train, y_train))

    # ------------------------------test the model-------------------------------------

    y_predicted = decision_tree.predict(X_test)
    misclassified_count = (y_test != y_predicted).sum()
    print('number of misclassified counts - ', misclassified_count)
    print('accuracy - ', accuracy_score(y_test, y_predicted))
    pass

    scores_based_on_depth(X_train, X_test, y_train, y_test)

# -------------- scores based on depth of tree----------------------------------------


def scores_based_on_depth(X_train, X_test, y_train, y_test):
    print('\n\n')
    print('depth\t\ttrain_score\t\ttest_score')
    for i in range(1, 13):
        decision_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=0)
        decision_tree.fit(X_train, y_train)
        y_predicted = decision_tree.predict(X_test)
        print(str(i) + '\t', decision_tree.score(X_train, y_train), '\t', accuracy_score(y_test, y_predicted))


if __name__ == '__main__':
    decision_tree()