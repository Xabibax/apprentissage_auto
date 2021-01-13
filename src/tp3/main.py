import inspect
from io import StringIO

import numpy as np
import pandas as pd
import pydot

from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split

import matplotlib.pyplot as plt


def write_clf_pdf(clf, output_name):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_pdf(output_name)


def function_name():
    return inspect.stack()[1][3]

class Perceptron:
    def __init__(self, nb_nodes=1, weight= [1]):
        self.

def neurologue():


    print(f"{function_name()} done.\n")


if __name__ == "__main__":

    neurologue()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
