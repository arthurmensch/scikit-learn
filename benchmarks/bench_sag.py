import time

from sklearn.datasets import fetch_rcv1
from sklearn.externals.joblib import delayed, Parallel, Memory
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.sag import get_auto_step_size
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import numpy as np

import matplotlib.pyplot as plt

class Callback:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.test_time = 0
        self.time = []
        self.train_score = []
        self.test_score = []
        self.accuracy = []
        self.start_time = time.clock()

    def __call__(self, lr):
        test_time = time.clock()
        y_pred = lr.predict_proba(self.X_train)
        train_score = lr.C * log_loss(self.y_train, y_pred, normalize=False)
        train_score += 0.5 * np.sum(lr.coef_ ** 2)
        print('Train score', train_score)
        y_pred = lr.predict_proba(self.X_test)
        test_score = lr.C * log_loss(self.y_test, y_pred, normalize=False)
        test_score += 0.5 * np.sum(lr.coef_ ** 2)
        print('Test score', test_score)
        accuracy = lr.score(self.X_test, self.y_test)
        print('Test accuracy', accuracy)
        self.accuracy.append(accuracy)
        self.train_score.append(train_score)
        self.test_score.append(test_score)
        self.test_time += time.clock() - test_time
        this_time = time.clock() - self.start_time - self.test_time
        self.time.append(this_time)


def fit_single(solver, X_train, X_test, y_train, y_test):
    callback = Callback(X_train, X_test, y_train, y_test)

    lr = LogisticRegression(solver=solver, multi_class='multinomial',
                            fit_intercept=True, tol=1e-10, max_iter=3,
                            verbose=True)
    # Private assignment
    lr._callback = callback

    lr.fit(X_train, y_train)
    return callback, lr


def exp():
    solvers = ['sag', 'saga']
    mem = Memory(cachedir='cache')

    rcv1 = fetch_rcv1()

    lbin = LabelBinarizer()
    lbin.fit(rcv1.target_names)

    X = rcv1.data
    y = rcv1.target
    y = lbin.inverse_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42,
                                                        stratify=None)

    cached_fit = mem.cache(fit_single)
    out = Parallel(n_jobs=2, mmap_mode=None)(
        delayed(cached_fit)(solver, X_train, X_test, y_train, y_test)
        for solver in solvers)

    callbacks, lrs = zip(*out)

    fig = plt.figure()
    ax = fig.add_subplot(131)

    ref = np.min(np.concatenate([np.array(callbacks[0].train_score),
                                np.array(callbacks[1].train_score)])) * 0.9

    for callback, solver in zip(callbacks, solvers):
        score = np.array(callback.train_score) / ref - 1
        ax.plot(callback.time, score, label=solver)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Training objective (relative to min)')
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax = fig.add_subplot(132)

    ref = np.min(np.concatenate([np.array(callbacks[0].test_score),
                                np.array(callbacks[1].test_score)])) * 0.9

    for callback, solver in zip(callbacks, solvers):
        score = np.array(callback.test_score) / ref - 1
        ax.plot(callback.time, score, label=solver)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Test objective (relative to min)')
    ax.set_yscale('log')

    ax = fig.add_subplot(133)

    for callback, solver in zip(callbacks, solvers):
        score = np.array(callback.accuracy)
        ax.plot(callback.time[1:], score[1:], label=solver)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Test accuracy')

    ax.legend()
    plt.show()

if __name__ == '__main__':
    exp()
