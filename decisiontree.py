# decisiontree.py
"""Predict Parkinson's disease based on dysphonia measurements using a decision tree."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pdb

import sklearn.metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix

ROOT = os.path.dirname(os.path.abspath(__file__))  # root directory of this code


def main():
    # Load data from relevant files
    xtrain = np.loadtxt(os.path.join(ROOT, "trainingdata.txt"), dtype=float, delimiter=',')
    ytrain = np.loadtxt(os.path.join(ROOT, "traininglabels.txt"), dtype=int)
    xtest = np.loadtxt(os.path.join(ROOT, "testingdata.txt"), dtype=float, delimiter=',')
    ytest = np.loadtxt(os.path.join(ROOT, "testinglabels.txt"), dtype=int)
    attributes = np.loadtxt(os.path.join(ROOT, "attributes.txt"), dtype=str)

    # Train a decision tree via information gain on the training data
    names = ["Positive", "Negative"]
    tree = DecisionTreeClassifier(criterion='entropy').fit(xtrain, ytrain)
    tree.predict(xtrain)

    # Test the decision tree
    train_predictions = tree.predict(xtrain)
    test_predictions = tree.predict(xtest)
    print(test_predictions)

    # Show the confusion matrix for test data
    train_cm = confusion_matrix(ytrain, train_predictions)
    test_cm = confusion_matrix(ytest, test_predictions)
    print("Training confusion matrix:")
    print(train_cm)
    print("Testing confusion matrix:")
    print(test_cm)

    # Compare training and test accuracy
    train_num_correct = 0
    for x in range(len(train_cm)):
        train_num_correct = train_num_correct + train_cm[x][x]
    test_num_correct = 0
    for x in range(len(test_cm)):
        test_num_correct = test_num_correct + test_cm[x][x]

    train_accuracy = train_num_correct / len(ytrain)
    test_accuracy = test_num_correct / len(ytest)
    print("Training accuracy: " + str(train_accuracy))
    print("Testing accuracy: " + str(test_accuracy))

    # Visualize the tree using matplotlib and plot_tree
    plot_tree(tree, feature_names=attributes, fontsize=4, filled=True, rounded=True, class_names=names)
    print(plt.show())


if __name__ == '__main__':
    main()
