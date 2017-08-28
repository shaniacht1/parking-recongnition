from __future__ import division
import numpy as np

class empca():
    def __init__(self,A, m, n, max_iter, max_err, k):
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
        if norm ==0:
            return v
        return v/norm

    def normc(self, V):
        return np.apply_along_axis(self.normalize, 0, V)

    def empca(self, k, max_iter, max_err):
        randomVector = self.normalize(np.random.randint(0, 256, (1,self.d)))
        c = np.zeros((self.N, 1), dtype=np.float)
        j=0
        while j<k:
            print 'while iter num ', j
            self.V[:,j] = randomVector
            i = 0
            while i < max_iter:
                print 'entering second while'
                u0 = self.V[:,j]
                sum  = np.zeros((1, self.d), dtype=np.float)
                for m in range(0, self.N):
                    c = np.dot(self.Mod_A[m,:], u0)
                    sum += np.dot(c, self.Mod_A[m,:])
                print 'after for'
                self.V[:,j]= self.normalize(sum.ravel())
                print 'after ravel'
                if i>10 and np.max(np.abs(self.V[:,j] - u0)) <= max_err:
                    break
                print 'after if'
                i += 1
            print 'after lala'
            self.Mod_A -= np.outer(np.dot(self.Mod_A, self.V[:,j]),  self.V[:,j].T)
            print 'blalb blabla fdg fdg '
            j += 1

        alpha = np.dot(self.V.T, self.A.T)
        print 'cha cha cha: ', np.cov(alpha)
        temp = np.cov(alpha)
        self.S, eigVe = np.linalg.eigh(np.dot(0.02, temp))
        print 'fdfd'
        print eigVe
        print self.V
        print self.S
        self.V = np.dot(self.V, eigVe).T


