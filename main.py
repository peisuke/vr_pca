import numpy as np
from vr_pca import *

n = 100
d = 30

X = np.random.rand(n, d)

m = n
r_h = (X **2).sum() / n
eta = 1 / (r_h * np.sqrt(n))

w = vr_pca(X, m, eta)
v = np.mean(X.T.dot(X).dot(w) / w)

print('VR_PCA')
print(w)
print(v)

u_, w_, _ = np.linalg.svd(X.T.dot(X))

print('SVD')
print(u_[:,0])
print(w_[0])
