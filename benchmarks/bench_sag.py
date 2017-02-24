import time

from sklearn.datasets import fetch_rcv1
from sklearn.externals.joblib import delayed, Parallel, Memory
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import numpy as np

import matplotlib.pyplot as plt

class Callback:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.test_time = 0
        self.time = []
        self.score = []
        self.start_time = time.clock()

    def __call__(self, lr):
        test_time = time.clock()
        y_pred = lr.predict_proba(self.X)
        score = lr.C * log_loss(self.y, y_pred, normalize=False)
        score += 0.5 * np.sum(lr.coef_ ** 2)
        print('Score', score)
        self.score.append(score)
        self.test_time += time.clock() - test_time
        this_time = time.clock() - self.start_time - self.test_time
        self.time.append(this_time)


rcv1 = fetch_rcv1()

lbin = LabelBinarizer()
lbin.fit(rcv1.target_names)

X = rcv1.data
X = X
y = rcv1.target
y = y
y = lbin.inverse_transform(y)
X = X[::10000]
y = y[::10000]

X_train, X_test, y_train, y_test = train_test_split(X, y)


def fit_single(solver):
    callback = Callback(X_train, y_train)

    lr = LogisticRegression(solver=solver, multi_class='multinomial',
                            fit_intercept=True, tol=1e-10, max_iter=20,
                            callback=callback, verbose=True)

    lr.fit(X_train, y_train)
    return callback, lr

solvers = ['sag', 'saga']

mem = Memory(cachedir='cache')
mem.clear()

cached_fit = mem.cache(fit_single)
out = Parallel(n_jobs=2)(
    delayed(cached_fit)(solver) for solver in solvers)

callbacks, lrs = zip(*out)

fig = plt.figure()
ax = fig.add_subplot(111)

for callback, solver in zip(callbacks, solvers):
    ax.plot(callback.time, callback.score, label=solver)
ax.legend()
plt.show()
