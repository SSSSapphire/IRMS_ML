import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy.testing import assert_almost_equal


np.set_printoptions(precision=2)
rng = np.random.RandomState(1)

W = rng.rand(2,2)
X_normal = rng.normal(scale=5, size=(2,20))
X_orig = W @ X_normal
X_mean = X_orig.mean(axis=1)[:,np.newaxis]
X = X_orig - X_mean
mean = X.mean(axis=1)

assert_almost_equal(0,mean)

#print('X.shape:',X.shape,'\n')
#print(X)
plt.scatter(X[0],X[1])
plt.show()
