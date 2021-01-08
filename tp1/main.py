import inspect
from io import StringIO

import pandas as pd
import pydot

from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split


def write_clf_pdf(clf, output_name):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph[0].write_pdf(output_name)


def function_name():
    return inspect.stack()[1][3]


def cabin_to_num(cabin):
    cabin = cabin.split(' ')[0]
    return int(str(ord(cabin[0]) - ord('A')) + cabin[1:])


def embarked_to_int(e):
    return int(ord(e) - ord('A'))


def maraichere():
    # Texture
    LISSE = 0
    RUGUEUX = 1

    # Etiquette
    POMME = "Pomme"
    ORANGE = "Orange"

    X = [(140, LISSE), (130, LISSE), (150, RUGUEUX), (170, RUGUEUX)]
    y = [POMME, POMME, ORANGE, ORANGE]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)

    unk_fruit = (170, LISSE)
    y_predict = clf.predict([unk_fruit])
    print(f"Let's predict the species of this fruit :\n"
          f"{unk_fruit}")
    print(f"The Decision Tree prediction is :\n"
          f"{'Orange' if y_predict == ORANGE else 'Pomme'}")

    write_clf_pdf(clf, "fruits.pdf")
    print(f"{function_name()} done.\n")


def fleuriste():
    from sklearn.datasets import load_iris

    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    random_state = 42
    max_leaf_nodes = 3

    clf = tree.DecisionTreeClassifier(random_state=random_state)
    clf = clf.fit(X_test, y_test)

    clf_score = clf.score(X_test, y_test)
    print(f"The score of the Decision Tree is :\n"
          f"{clf_score}")

    write_clf_pdf(clf, "no_max_leaf_nodes.pdf")

    clf = tree.DecisionTreeClassifier(random_state=random_state, max_leaf_nodes=max_leaf_nodes)
    clf = clf.fit(X_train, y_train)

    clf_score = clf.score(X_test, y_test)
    print(f"The score of the Decision Tree with limited leaf nodes is :\n"
          f"{clf_score}")

    write_clf_pdf(clf, "max_leaf_nodes_to_3.pdf")

    score = cross_val_score(clf, X_test, y_test, cv=5)
    print(f"The cross val scores for the Decision Tree is :\n"
          f"{score}")
    print(f"The description of the cross val scores for the Decision Tree is :\n"
          f"{pd.DataFrame(score).describe()}")

    print(f"{function_name()} done.\n")


def titanic():
    df = pd.read_csv('titanic.csv', sep=',')
    # print(df)

    interesting_col = ['Survived', 'Pclass', 'Sex', 'Age', 'Cabin', 'Embarked']

    df = df[interesting_col]
    # print(df)

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    # print(df)

    df['Embarked'] = df['Embarked'].map(embarked_to_int, na_action='ignore')
    # print(df)

    df['Cabin'] = df['Cabin'].map(cabin_to_num, na_action='ignore')

    df = df.dropna()
    # print(df)
    interesting_col.remove('Survived')
    X = df[interesting_col]
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    clf_score = clf.score(X_test, y_test)
    print(f"The clf score is :\n"
          f"{clf_score}")

    y_predict = clf.predict(X_test)

    cfm = confusion_matrix(y_test, y_predict)
    conf_mat = pd.DataFrame(
        cfm,
        columns=['Mort prevue', 'Survie prevue'],
        index=['Mort', 'Survivant']
    )

    total_col = pd.Series(
        conf_mat.sum(axis=0, numeric_only=True).values,
        index=['Mort prevue', 'Survie prevue'],
        name='Total'
    )

    conf_mat = conf_mat.append(total_col)

    total_row = conf_mat.sum(axis=1, numeric_only=True).values
    conf_mat['Total'] = total_row
    print(conf_mat)

    print(f"{function_name()} done.\n")


if __name__ == "__main__":
    maraichere()
    fleuriste()
    titanic()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
