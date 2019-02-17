"""The idea behind this code is you need to iterate over the step sizes and for every bias from the range of bias
apply all transformations to weight vector find the suitable weight and bias and then as [convex approach] decrease
the weight and proceed further """

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('ggplot')


class SVM(object):
    def __init__(self):
        self.colors = {1: 'r', -1: 'b'}
        self.weight = None
        self.bias = None
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, data):
        optima_dict = dict()
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]
        flattened_data = np.array(list())
        for Yi in data:
            flattened_data = np.append(flattened_data, data[Yi])

        max_in_data, min_in_data = max(flattened_data), min(flattened_data)
        step_sizes = [0.1 * max_in_data, 0.01 * max_in_data, 0.001 * max_in_data]

        local_optima = max_in_data * 10

        # step and multiplier for bias
        bias_range_multiple = 5
        bias_multiple = 5

        '''1. bias is amount of the shift we can allow for the hyperplane, initialize it random value but it can be 
        zero also 
           2. Now we need to satisfy yi(x.w)+b>=1 for all training dataset such that ||w|| is minimum for this 
        we will start with random w, and try to satisfy it with making b bigger and bigger 
           3. Convex optimization is 
        followed in finding the global optimal, so step is added only when we need to move forward '''

        for step in step_sizes:
            weight_vector = np.array([local_optima, local_optima])
            is_optimized = False

            while not is_optimized:

                for bias in np.arange(-1 * max_in_data * bias_range_multiple, max_in_data * bias_range_multiple,
                                      step * bias_multiple):

                    for transformation in transforms:
                        transformed_wt_vector = weight_vector * transformation
                        found_option = True
                        for Yi in data:
                            for Xi in data[Yi]:
                                if not Yi * (np.dot(transformed_wt_vector, Xi) + bias) >= 1:
                                    found_option = False

                        if found_option:
                            optima_dict[np.linalg.norm(transformed_wt_vector)] = [transformed_wt_vector, bias]

                print(weight_vector, len(optima_dict))
                if weight_vector[0] < 0:
                    is_optimized = True
                    print("optimized a step")
                else:
                    weight_vector -= step

            sorted_opt_list = sorted([item for item in optima_dict])

            # select optimal weight vector and bias
            opt_dict = optima_dict[sorted_opt_list[0]]
            self.weight = opt_dict[0]
            self.bias = opt_dict[1]

            local_optima = opt_dict[0][0] + step * 2

    def make_prediction(self, feature):
        classification = np.sign(np.dot(np.array(feature), self.weight) + self.bias)
        return classification


data_dict = {-1: np.array([[1, 7], [2, 8], [3, 8]]), 1: np.array([[5, 1], [6, -1], [7, 3]])}
svm = SVM()  # Linear Kernel
svm.fit(data=data_dict)
print(svm.weight, svm.bias)
# svm.visualize()
prediction = svm.make_prediction([1, 7])
print(prediction)
