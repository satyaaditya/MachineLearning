'''
AIM - The aim of this attempt is to predict if the client will subscribe (yes/no) to a term deposit, by building a classification model using Decision Tree.
'''
# complete info -https://www.kaggle.com/shirantha/bank-marketing-data-a-decision-tree-approach?scriptVersionId=2248246
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import numpy as np


def load_csv():
    data = pd.read_csv('datasets/bank.csv')
    print('--------------- data preview\n', data.head())
    return data


def check_for_null_values_in_data(data):
    print('---------------check for null values')
    print(data[data.isnull().any(axis=1)].count())


def data_exploration(data):
    # check which category employee are having most deposits
    jobs = ['management', 'blue-collar', 'technician', 'admin.', 'services', 'retired', 'self-employed', 'student',
            'unemployed', 'entrepreneur', 'housemaid', 'unknown']
    print('\n\ncheck which category employee are having most deposits --')
    for job in jobs:
        print('{:15}:{:5}'.format(job, len(data[(data['deposit'] == 'yes') & (data['job'] == job)])))

    print('\n\nemploy count for each job --')
    print(data['job'].value_counts())


def data_visualization(data, feature):
    seaborn.boxplot(x=data[feature])
    plt.show()
    seaborn.distplot(data[feature], bins=100)
    plt.show()


def data_pre_processing(bank_data):
    # merge similar category jobs
    bank_data['job'] = bank_data['job'].replace(['management', 'admin.'], 'white-collar')
    bank_data['job'] = bank_data['job'].replace(['services', 'housemaid'], 'pink-collar')
    bank_data['job'] = bank_data['job'].replace(['retired', 'student', 'unemployed', 'unknown'], 'other')
    print(bank_data.head())

    # convert unknow outcome to other
    bank_data['poutcome'] = bank_data['poutcome'].replace(['unknown'], 'other')
    # drop contact
    bank_data.drop('contact', axis=1, inplace=True)
    # convert string input to integer
    bank_data['default'], _ = pd.factorize(bank_data['default'])
    bank_data['housing'], _ = pd.factorize(bank_data['housing'])
    bank_data['loan'], _ = pd.factorize(bank_data['loan'])
    bank_data['deposit'], _ = pd.factorize(bank_data['deposit'])
    bank_data.drop('month', axis=1, inplace=True)
    bank_data.drop('day', axis=1, inplace=True)

    # replace poutcome = -1 with large value as in future operation it shoudn't affect
    bank_data.loc[bank_data['pdays'] == -1, 'pdays'] = 10000
    bank_data['recent_pdays'] = np.where(bank_data['pdays'], 1 / bank_data.pdays, 1 / bank_data.pdays)
    bank_data.drop('pdays', axis=1, inplace=True)

    # consider data frame with remaining attr like marital etc.. as dummies
    bank_data = pd.get_dummies(bank_data, columns=['job', 'marital', 'education', 'poutcome'],
                               prefix=['job', 'marital', 'education', 'poutcome'])
    print(bank_data.head())
    return bank_data


def scores_based_on_depth(X_train, X_test, y_train, y_test):
    print('\n\n')
    print('depth\t\ttrain_score\t\ttest_score')
    for i in range(1, 13):
        decision_tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=i, random_state=0)
        decision_tree.fit(X_train, y_train)
        y_predicted = decision_tree.predict(X_test)
        print(str(i) + '\t', decision_tree.score(X_train, y_train), '\t', accuracy_score(y_test, y_predicted))


def decision_tree():
    data = load_csv()
    check_for_null_values_in_data(data)
    # data_visualization(data, 'age')
    data_exploration(data)
    data = data_pre_processing(data)
    # drop target variable
    independent_data = data.drop('deposit', axis=1)
    dependent_data = data.deposit
    X_train, X_test, y_train, y_test = train_test_split(independent_data, dependent_data, test_size=0.3)
    scores_based_on_depth(X_train, X_test, y_train, y_test)
    pass


if __name__ == '__main__':
    decision_tree()
