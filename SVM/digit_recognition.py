import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

train_data = pd.read_csv('datasets/dgt_rcg_train.csv')
test_data = pd.read_csv('datasets/dgt_rcg_test.csv')

# print('train data shape', train_data.shape)
# print('test data shape', test_data.shape)

# print(train_data.head())

images = train_data.iloc[0:5000, 1:]
labels = train_data.iloc[0:5000, :1]

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

plt.hist(train_images.iloc[49])
# plt.show()

classifier = svm.SVC()
classifier.fit(train_images, train_labels.values.ravel())
print(classifier.score(test_images, test_labels))
pass
train_images[train_images > 0] = 1
test_images[test_images > 0] = 1

clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
print(clf.score(test_images, test_labels))

test_data[test_data > 0] = 1
results = classifier.predict(test_data)
print(results)
