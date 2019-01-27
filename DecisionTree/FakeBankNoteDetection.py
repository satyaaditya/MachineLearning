from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn import tree
import pandas as pd


def load_csv():
    data = pd.read_csv('datasets/banknote_authentication.csv')
    print('--------------- data preview\n', data.head())
    return data


def check_for_null_values_in_data(data):
    print('---------------check for null values')
    print(data[data.isnull().any(axis=1)].count())


def use_kfold(independent_data, dependent_data):
    """ kfold split k-1 parts for training and 1 for test, so
        you need to give complete data without split, its different from train_test_split
     """
    decision_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
    kf = KFold(n_splits=10)
    scores = cross_val_score(decision_tree, independent_data, dependent_data, cv=kf)
    print('kfold -')
    print('accuracy - ', scores.mean())


def decision_tree(data):
    independent_data = data.drop(columns=[data.columns[-1]])
    dependent_data = data.iloc[:, -1]
    # print(dependent_data.head())
    X_train, X_test, y_train, y_test = train_test_split(independent_data, dependent_data, test_size=0.3)
    decision_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
    decision_tree.fit(X_train, y_train)
    y_predicted = decision_tree.predict(X_test)
    print(decision_tree.score(X_train, y_train), '\t', accuracy_score(y_test, y_predicted))
    use_kfold(independent_data, dependent_data)  # try using kfold


if __name__ == '__main__':
    data = load_csv()
    check_for_null_values_in_data(data)
    decision_tree(data)
