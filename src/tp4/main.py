import inspect
import random
from io import StringIO

import numpy as np
import pydot

from sklearn import tree

import gym


def write_clf_pdf(clf, output_name):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_pdf(output_name)


def function_name():
    return inspect.stack()[1][3]


def equilibriste():
    env = gym.make('CartPole-v0')
    env.reset()
    for _ in range(200):
        env.render()
        env.step(env.action_space.sample())  # take a random action

    env.close()

    print(f"{function_name()} done.\n")


if __name__ == "__main__":
    equilibriste()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
