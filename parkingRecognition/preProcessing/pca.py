from PIL import Image
from numpy import *
from pylab import plt

def pca(X, m, n):
  # get dimensions
  num_data,dim = X.shape

  # center data
  mean_X = X.mean(axis=0)
  X = X - mean_X

  if dim>num_data:
    M = dot(X,X.T)
    e,EV = linalg.eigh(M)
    tmp = dot(X.T,EV).T
    V = tmp[::-1]
    S = sqrt(e)[::-1]
    for i in range(V.shape[1]):
      V[:,i] /= S
  else:
    U,S,V = linalg.svd(X)
    V = V[:num_data]
  return V,S,mean_X