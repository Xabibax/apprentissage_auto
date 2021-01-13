import inspect
from io import StringIO

import numpy as np
import pandas as pd
import pydot

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

import matplotlib.pyplot as plt


def write_clf_pdf(clf, output_name):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_pdf(output_name)


def function_name():
    return inspect.stack()[1][3]


def jardiniere():

    iris = load_iris()
    # print(iris.keys())
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y)


    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)

    cfm = confusion_matrix(y_test, y_predict)
    conf_mat = pd.DataFrame(
        cfm,
        columns=list(map(lambda t: f"{t} prévue", iris.target_names)),
        index=iris.target_names
    )
    print(conf_mat)

    clf_rep = classification_report(y_test, y_predict)
    print(clf_rep)

    err = []
    for i in range(1, 40):
        clf_i = KNeighborsClassifier(n_neighbors=i)
        clf_i.fit(X_train, y_train)

        y_predict_i = clf_i.predict(X_test)

        err.append(np.mean(y_predict_i != y_test))

    plt.figure()
    plt.plot(range(1, 40), err)
    plt.show()

    print(f"{function_name()} done.\n")


def magicien():

    # X, y = make_blobs(n_samples=100, centers=2, random_state=6)
    # X, y = make_circles(n_samples=400, factor=.3, noise=.05)
    X, y = make_moons(n_samples=400, noise=.3)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidths=1, facecolors='none',
               edgecolors='k')

    plt.show()
    print(f"{function_name()} done.\n")


if __name__ == "__main__":
    # jardiniere()
    magicien()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
