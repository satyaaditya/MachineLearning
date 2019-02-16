import numpy as np
from matplotlib import pyplot as plot

data = np.array([
    [-2, 4, -1],
    [4, 1, -1],
    [1, 4.2, -1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],

])

data_labels = np.array([-1, -1, -1,  1, 1, 1])

# plot these variables on graph

for index, data_point in enumerate(data):
    if index < 3:
        plot.scatter(data_point[0], data_point[1], s=120, marker='_', linewidths=2)
    else:
        plot.scatter(data_point[0], data_point[1], s=120, marker='+', linewidths=2)

# draw a sample hyperplane, by naive guess
plot.plot([-1, 5], [5, 1.5])
plot.show()


def perform_classification(data, data_lablels):
    # this includes selecting our optimal hyper plane and classifying data accordingly
    iterations = 10000
    learning_rate = 1
    # svm weight vector
    w = np.zeros(len(data[0]))
    errors_in_classification = []
    for iteration in range(1, iterations + 1):
        error = 0
        for i, data_point in enumerate(data):
            if data_lablels[i] * np.dot(w, data[i]) < 1:
                w = w + learning_rate * ((data_lablels[i] * data[i]) + (-2 * (1 / iteration) * w))
                error = 1
            else:
                w = w + (learning_rate * (-2 * (1 / iteration) * w))
        errors_in_classification.append(error)

    plot.plot(errors_in_classification, '|')
    plot.ylim(0.5, 1.5)
    plot.axes().set_yticklabels([])
    plot.xlabel('Epoch')
    plot.ylabel('Misclassified')
    plot.show()
    return w


w = perform_classification(data, data_labels)
for d, sample in enumerate(data):
    # Plot the negative samples
    if d < 3:
        plot.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plot.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# # Add our test samples
# plot.scatter(2, 2, s=120, marker='_', linewidths=2, color='yellow')
#
# plot.scatter(4, 3, s=120, marker='+', linewidths=2, color='black')
#

# Print the hyperplane calculated by svm_sgd()
x2 = [w[0], w[1], -w[1], w[0]]
x3 = [w[0], w[1], w[1], -w[0]]

x2x3 = np.array([x2, x3])
X, Y, U, V = zip(*x2x3)
ax = plot.gca()
ax.quiver(X, Y, U, V, scale=1, color='red')
plot.show()