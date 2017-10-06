import time
import numpy as np
from vr_pca import *

n = 10000
d = 3000

X = np.random.rand(n, d)

m = n
r_h = (X **2).sum() / n
eta = 1 / (r_h * np.sqrt(n))

start_time = time.time()
w = vr_pca(X, m, eta)
s = time.time() - start_time
v = np.mean(X.T.dot(X).dot(w) / w)

print('VR_PCA: %f' % s)
print(w[:10])
print(v)

start_time = time.time()
u_, w_, _ = np.linalg.svd(X.T.dot(X))
s = time.time() - start_time

print('SVD: %f' %s)
print(u_[:,0][:10])
print(w_[0])
