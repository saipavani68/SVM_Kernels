import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm
from sklearn import preprocessing
from sklearn.datasets import make_blobs


fignum= 1
n_samples = 100
X, y = make_blobs(n_samples=n_samples, centers=2, random_state=6)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

for kernel in ('linear', 'poly', 'rbf'):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)
    plt.figure()
    plt.clf()
    y_training_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    training_accuracy = metrics.accuracy_score(y_train, y_training_pred)
    testing_accuracy = metrics.accuracy_score(y_test, y_test_pred)
    training_error = 1 - training_accuracy
    testing_error = 1 - testing_accuracy
    print("Training set error rate " + kernel + ": " + str(training_error))
    print("Testing set error rate" + kernel + ": " + str(testing_error))
    plt.scatter(X[:,0], X[:,1], c=y, zorder=10, s= 20, cmap=plt.cm.Paired, edgecolor='k')
    
    # Circling out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10, edgecolor='k')
    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.title(kernel)
plt.show()