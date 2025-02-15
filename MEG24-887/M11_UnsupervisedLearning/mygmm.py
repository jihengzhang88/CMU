#adopted from https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_sin.html#sphx-glr-auto-examples-mixture-plot-gmm-sin-py

import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from scipy import linalg

from sklearn import mixture


plt.rcParams['figure.dpi'] = 200

color_iter = itertools.cycle(["red", "blue", "cornflowerblue", "gold", "darkorange"])


def plot_results(X, Y, means, covariances, title):
    splot = plt.subplot(1, 1, 1)
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], s=8, color=color, alpha = 0.5)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_alpha(0.5)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.title(title)
    plt.xticks(())
    plt.yticks(())


def plot_samples(X, Y, n_components, title):
    for i, color in zip(range(n_components), color_iter):
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y == i):
            continue
        plt.scatter(X[Y == i, 0], X[Y == i, 1], s=8, color=color, alpha = 0.5)

    plt.title(title)
    plt.xticks(())
    plt.yticks(())


# Parameters
n_samples = 100

# Generate data
X, y = make_blobs(n_samples = 2000, n_features = 2, centers = 5)
plt.rcParams['figure.dpi'] = 200
plt.scatter(X[:,0], X[:,1], c = 'black', alpha = 0.5)
plt.show()

mean = [0, 0]
cov = [[1, 0], [0, 1]]  # diagonal covariance
x, y = np.random.multivariate_normal(mean, cov, 100).T
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
X1 = np.hstack((x,y))

mean = [5, 0]
cov = [[1, 0], [0, 1]]  # diagonal covariance
x, y = np.random.multivariate_normal(mean, cov, 100).T
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
X2 = np.hstack((x,y))

mean = [10, 0]
cov = [[1, 0], [0, 1]]  # diagonal covariance
x, y = np.random.multivariate_normal(mean, cov, 100).T
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
X3 = np.hstack((x,y))

mean = [2.5, 5]
cov = [[2, -1], [-3, 1]]  # diagonal covariance
x, y = np.random.multivariate_normal(mean, cov, 100).T
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
X4 = np.hstack((x,y))

mean = [7.5, 5]
cov = [[4, -4], [0, 8]]  # diagonal covariance
x, y = np.random.multivariate_normal(mean, cov, 100).T
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
X5 = np.hstack((x,y))

X = np.concatenate((X1, X2, X3, X4, X5), axis=0)
print(X.shape)

mean = [0, 0]
cov = [[2,-2], [-2, 0.2]]  # diagonal covariance
x, y = np.random.multivariate_normal(mean, cov, 300).T
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
X6 = np.hstack((x,y))

mean = [0,0]
cov = [[0.1, 0], [0, 1]]  # diagonal covariance
x, y = np.random.multivariate_normal(mean, cov, 300).T
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
X7 = np.hstack((x,y))

X = np.concatenate((X6,X7), axis=0)
print(X.shape)



plt.rcParams['figure.dpi'] = 200
plt.scatter(X[:,0], X[:,1], c = 'black', alpha = 0.5)
plt.show()


# Fit a Gaussian mixture with EM using ten components
#gmm = mixture.GaussianMixture(n_components=5, covariance_type="full", max_iter=100).fit(X)
gmm = mixture.GaussianMixture(n_components=2, covariance_type="full", max_iter=100).fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, "GMM with Expectation-maximization")

plt.show()