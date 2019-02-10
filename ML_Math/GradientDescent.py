import pandas as pd


def load_data():
    data = pd.read_csv('datasets/Admission_Predict.csv')
    return [data['TOEFL Score'], data['GRE Score']]


def load_data1():
    data = pd.read_csv('datasets/sample.csv')
    return [data['a'], data['b']]


def error_measure(b, m, data):
    total_y = 0
    for iteration in range(len(data[0])):
        total_y += (data[1][iteration] - (m * data[0][iteration] + b)) ** 2
    return total_y / float(len(data[0]))


def calculate_gradient_descent(data, b, m, learning_rate=0.0001):
    """ derivative gives us the slope and this stub gives avg b, m. This method is being called over 1000 iterations
    to get a minima """
    b_final = 0
    m_final = 0
    N = float(len(data[0]))
    for i in range(len(data[0])):
        b_final += -(2 / N) * (data[1][i] - (m * data[0][i] + b))  # sqr(y - h(x)) / n  derivative wrt b
        m_final += -(2 / N) * data[0][i] * (data[1][i] - (m * data[0][i] + b))  # wrt x
    b_final = b - (b_final * learning_rate)
    m_final = m - (m_final * learning_rate)
    return [b_final, m_final]
    pass


def gradient_descent_runner(data, starting_b, starting_m, num_iterations):
    b = starting_b
    m = starting_m
    learning_rate = 0.0001
    for i in range(num_iterations):
        b, m = calculate_gradient_descent(data, b, m, learning_rate)
    return [b, m]


def run_gradient_descent():
    data = load_data()
    initial_b = 0
    initial_m = 0
    iterations = 1000
    print('error measure initially for b = {}, m = {}, error = {}'.format(initial_b, initial_m,
                                                                          error_measure(initial_b, initial_m, data)))
    print('gradient descent running.....')

    b, m = gradient_descent_runner(data, initial_b, initial_m, iterations)

    print('error measure after for b = {}, m = {}, error = {}'.format(b, m,
                                                                      error_measure(b, m, data)))


if __name__ == '__main__':
    run_gradient_descent()
