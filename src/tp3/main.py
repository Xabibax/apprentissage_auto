import inspect
import random
from io import StringIO

import numpy as np
import pydot

from sklearn import tree


def write_clf_pdf(clf, output_name):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_pdf(output_name)


def function_name():
    return inspect.stack()[1][3]


class Perceptron:
    def __init__(self, weight=[1]):
        self.nb_nodes = len(weight)
        self.weight = weight

    def prod_scal(self, input):
        return np.vdot(self.weight, input)

    def activate(self, sum):
        return 1 if sum >= 0 else -1

    def update_weight(self, X, e):
        for i in range(self.nb_nodes):
            # wi = wi + e ∗ xi
            self.weight[i] += e * X[i]

    def predict(self, X, y):
        count = 0
        MAX_LOOP = 999
        while True:
            predictions, errors = [[], []]
            for i in range(len(X)):
                predictions.append(self.activate(self.prod_scal(X[i])))
                errors.append(y[i] - predictions[i])
                self.update_weight(X[i], errors[i])
            count += 1
            if all(e == 0 for e in errors):
                break
            elif 0 < MAX_LOOP < count:
                print(f"Max loop reached!")
                break

        return predictions, errors, self


def generate_training_set(num_points):
    x = [random.randint(0, 50) for _ in range(num_points)]
    y = [random.randint(0, 50) for _ in range(num_points)]
    data, target = [[], []]
    for x, y in zip(x, y):
        data.append((x, y))
        if x <= 45 - y:
            target.append(1)
        elif x > 45 - y:
            target.append(-1)
    return data, target


def neurologue():
    X, y = [[[0, 3], [3, 0], [0, -3], [-3, 0]], [1, -1, -1, 1]]
    perceptron = Perceptron([0] * len(X[0]))
    predictions, errors, perceptron = perceptron.predict(X, y)
    print(f"X is : {X}, \n"
          f"y is : {y}, \n"
          f"Predictions are : {predictions}, \n"
          f"Errors are : {errors}, \n"
          f"Weights are : {perceptron.weight}")

    print('--' * 40)

    X, y = generate_training_set(5)
    perceptron = Perceptron([0] * len(X[0]))
    predictions, errors, perceptron = perceptron.predict(X, y)
    print(f"X is : {X}, \n"
          f"y is : {y}, \n"
          f"Predictions are : {predictions}, \n"
          f"Errors are : {errors}, \n"
          f"Weights are : {perceptron.weight}")

    print(f"{function_name()} done.\n")


if __name__ == "__main__":
    neurologue()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
