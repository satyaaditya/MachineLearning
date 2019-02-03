# we need to guess wheter the individuals from the test dataset had survived or not
import copy

import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np

def load_csv():
    train_data = pd.read_csv('datasets/titanic_train.csv')
    test_data = pd.read_csv('datasets/titanic_test.csv')
    return [train_data,test_data]

def check_null_values(data):
   print(data.isnull().sum())

def data_preproccessing(train_data, test_data):
    # convert cabin to
    check_null_values(train_data)
    complete_data = [train_data, test_data]

    train_data['Has_Cabin'] = train_data['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
    test_data['Has_Cabin'] = test_data['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

    # family size siblings or spouse / parent or children + yourself :)
    for dataset in complete_data:
        dataset['Family_Size'] = dataset['SibSp'] + dataset['Parch'] + 1

    # keep this key to remove sibsp and parch keys
    for dataset in complete_data:
        dataset['isAlone'] = 0
        dataset.loc[dataset['Family_Size'] == 1, 'isAlone'] = 1

    # Remove all NULLS in the Embarked column
    for dataset in complete_data:
        dataset['Embarked'] = dataset['Embarked'].fillna('S')

    for dataset in complete_data:
        dataset.loc[dataset['Fare'].isnull()] = train_data['Fare'].median()

    # replace null values in age
    for dataset in complete_data:
        avg_age = dataset['Age'].median()
        std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        random_list_of_avg_age = np.random.randint(avg_age - std, avg_age + std, size=age_null_count)
        dataset.loc[np.isnan(dataset['Age']), 'Age'] = random_list_of_avg_age
        dataset['Age'] = dataset['Age'].astype(int)

    import re
    def get_title(name):
        try:
            title_search = re.search(' ([A-Za-z]+)\.', name)
            # If the title exists, extract and return it.
            if title_search:
                return title_search.group(1)
            return ""

        except Exception:
            print(name)

    for dataset in complete_data:
        dataset['Title'] = dataset['Name'].apply(get_title)

    # replace relative titles with single title
    for dataset in complete_data:
        dataset['Title'] = dataset['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    for dataset in complete_data:
        dataset['Sex'], _ = pd.factorize(dataset['Sex'])
        dataset['Title'], _ = pd.factorize(dataset['Title'])
        dataset['Embarked'], _ = pd.factorize(dataset['Embarked'])

        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

        # Mapping Age
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
    train_data = train_data.drop(drop_elements, axis=1)
    test_data = test_data.drop(drop_elements, axis=1)

        # print(train_data.head())

    return [train_data, test_data]
    pass


def compare_features(train_data, test_data):
    '''comapare features using group by'''

    # mean is percentage of people survived
    # count is total count of that category
    # sum is number of people survived
    print(train_data[['Survived', 'Title']].groupby(['Title'], as_index=True).agg(['mean', 'count', 'sum']))
    print(train_data[['Survived', 'Sex']].groupby(['Sex'], as_index=True).agg(['mean', 'count', 'sum']))
    gini_impurity_total = gini_impurity(342, 891)
    gini_impurity_male = gini_impurity(109, 577)
    gini_impurity_female = gini_impurity(233, 314)
    print('gini impurity for survival vs total - ', gini_impurity_total)
    print('gini impurity for sex:male - ', gini_impurity_male)
    print('gini impurity for sex:female - ', gini_impurity_female)
    # Gini Impurity decrease if node splited for observations with Title == 1 == Mr
    men_weight = 577 / 891
    women_weight = 314 / 891
    weighted_gini_impurity_sex_split = (gini_impurity_male * men_weight) + (gini_impurity_female * women_weight)

    sex_gini_decrease = weighted_gini_impurity_sex_split - gini_impurity_total
    print(sex_gini_decrease)

    # similarly do gini decrease for mr title vs other titles and you will find out this has more decrease than sex_gini_decrease

def gini_impurity(survived_count, total_count):
    surival_prob = survived_count/total_count
    non_survival_prob = 1 - surival_prob
    mislabelling_survival_prob = surival_prob * non_survival_prob
    mislabelling_not_survival_prob = non_survival_prob * surival_prob
    return mislabelling_not_survival_prob + mislabelling_survival_prob


def randomForest(train, test):
    y_train = train['Survived']
    x_train = train.drop(['Survived'], axis=1).values
    x_test = test.values
    y_test = test['Survived']
    rf = RandomForestClassifier(criterion='gini',
                                n_estimators=700,
                                min_samples_split=10,
                                min_samples_leaf=1,
                                max_features='auto',
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)
    rf.fit(x_train, y_train)
    predictions = rf.predict(x_test)
    print("%.4f" % rf.oob_score_)
# def random_forest

if __name__ == '__main__':
    train_data, test_data = load_csv()
    original_train_data = copy.deepcopy(train_data)
    train_data, test_data = data_preproccessing(train_data, test_data)
    compare_features(train_data, test_data)
    randomForest(train_data, test_data)
