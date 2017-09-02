from __future__ import division
import numpy as np
from pylab import *
from pylab import plt
import scipy.linalg as la

class empca():
    def __init__(self,A, max_iter, max_err, k, m, n):
        # A = A.T
        self.N = A.shape[0]
        self.d = A.shape[1]
        print 'dd: ', self.d,' ' ,self.N
        self.mean_A = A.mean(axis=0)
        self.A = A - self.mean_A
        self.Mod_A = self.A.copy()
        # self.U = np.zeros((A.shape[0], k), dtype=np.float)     #matrix of size 50xk
        self.V = np.zeros((A.shape[1], k), dtype=np.float)    #matrix of size 2000000xk
        self.empca(k, max_iter, max_err)

    def normalize(self, v):
        norm = np.linalg.norm(v)
        # print 'norm: ', norm
        # print 'v: ', v
        if norm ==0:
            return v
        return v/norm

    def normc(self, V):
        return np.apply_along_axis(self.normalize, 0, V)

    def empca(self, k, max_iter, max_err):
        # randomVector = self.normalize(np.random.randint(0, 256, (1,self.d)))
        # c = np.zeros((self.N, 1), dtype=np.float)
        j=0
        while j<k:
            print 'while iter num ', j
            self.V[:,j] = np.random.randint(-256, 256, (1,self.d))
            # self.V[:, j] = self.normalize(np.random.randint(0, 256, (1, self.d)))
            # print 'random vector: ', self.V[:,j]
            # print 'random vector chosen'
            i = 0
            while i < max_iter:
                print 'entering second while: ', i
                u0 = self.V[:,j].copy()
                sum  = np.zeros((1, self.d), dtype=np.float)
                for m in range(0, self.N):
                    c = np.dot(self.Mod_A[m,:], u0)
                    sum += np.dot(self.Mod_A[m,:], c)
                    # print 'mod_a ', self.Mod_A[m,:]
                    # print 'c ', c
                    # print 'sum: ', sum
                self.V[:,j]= self.normalize(sum.ravel())
                # print 'sum after:  ', sum
                # print 'uo after: ' ,u0
                # print 'vj after: ', self.V[:,j]
                if i>10 and np.max(np.abs(self.V[:,j] - u0)) <= max_err:
                    print 'abs', np.max(np.abs(self.V[:,j] - u0))
                    # print self.V[:,j], ' vvv ' , u0
                    break
                i += 1
            # should reorth the found vector?
            self.Mod_A -= np.outer(np.dot(self.Mod_A, self.V[:,j]),  self.V[:,j].T)
            j += 1

        self.V = la.orth(self.V)
        alpha = np.dot(self.V.T, self.A.T)
        temp = np.cov(alpha)
        self.S, eigVe = np.linalg.eigh(np.dot(0.02, temp))
        # print 'fdfd'
        # print self.V
        # print self.S
        self.V = np.dot(self.V, eigVe).T


