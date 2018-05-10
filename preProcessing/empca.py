from __future__ import division
import numpy as np
from pylab import *
import scipy.linalg as la

class empca():
    def __init__(self,A, max_iter, max_err, k, m, n):
        self.N = A.shape[0]
        self.d = A.shape[1]
        self.mean_A = A.mean(axis=0)
        self.A = A - self.mean_A
        self.Mod_A = self.A.copy()
        self.V = np.zeros((A.shape[1], k), dtype=np.float)    #matrix of size dxk
        self.empca(k, max_iter, max_err)

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm ==0:
            return v
        return v/norm

    def normc(self, V):
        return np.apply_along_axis(self.normalize, 0, V)

    def empca(self, k, max_iter, max_err):
        j=0
        while j<k:
            print 'while iteration num ', j
            self.V[:,j] = np.random.randint(-256, 256, (1,self.d))
            i = 0
            while i < max_iter:
                print 'entering second while: ', i
                u0 = self.V[:,j].copy()
                sum  = np.zeros((1, self.d), dtype=np.float)
                for m in range(0, self.N):
                    c = np.dot(self.Mod_A[m,:], u0)
                    sum += np.dot(self.Mod_A[m,:], c)
                self.V[:,j]= self.normalize(sum.ravel())
                if i>10 and np.max(np.abs(self.V[:,j] - u0)) <= max_err:
                    break
                i += 1
            self.Mod_A -= np.outer(np.dot(self.Mod_A, self.V[:,j]),  self.V[:,j].T)
            j += 1

        self.V = la.orth(self.V)
        alpha = np.dot(self.V.T, self.A.T)
        temp = np.cov(alpha)
        self.S, eigVe = np.linalg.eigh(np.dot(0.02, temp))
        self.V = np.dot(self.V, eigVe).T


