from __future__ import division
import numpy as np
from scipy import stats

class rpca():
    def __init__(self, A, m, n, max_iter, max_err, k, w):
        self.N = A.shape[0]
        self.d = A.shape[1]
        self.mean_A = A.mean(axis=0)
        self.A = A - self.mean_A
        self.Mod_A = self.A.copy()
        self.V = np.zeros((self.d, k), dtype=np.float)  # matrix of size 2000000xk
        self.w = w
        print 'asdasdasdasdasd'
        print w
        if self.w is None:
            print 'ain\'t sunshine when she\'s gone'
            self.w = np.ones((1, self.N), dtype=np.float)
        self.rpca(k, max_iter, max_err)

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def normc(self, V):
        return np.apply_along_axis(self.normalize, 0, V)

    def normr(self, V):
        return np.apply_along_axis(self.normalize, 1, V)

    def mu(self, w, a, data, percent):
        times = np.multiply(a*w,data.T).T
        #w*a*data
        return stats.trim_mean(times, percent)

    def rpca(self, k, max_iter, max_err):
        # mod_a = u
        # self.Mod_A = normr(self.Mod_A)
        a = np.zeros((1, self.N), dtype=np.float)
        j = 0
        while j < k:
            self.Mod_A = self.normr(self.Mod_A)
            print 'while iter num ', j
            i = 0
            q_cur = self.normalize(np.random.randint(0, 256, (1, self.d)))
            while i < max_iter:
                print 'entering second while'
                q_prev = q_cur
                for n in range(0, self.N - 1):
                    print 'nannananan'
                    temp1 = self.Mod_A[n , :].reshape(self.d, 1)
                    print 'a'
                    print temp1.shape
                    print q_prev.shape
                    temp2 = np.dot(q_prev,temp1)
                    print 'b'
                    a[0,n] = np.sign(temp2)
                    print 'c'
                    print 'd'
                    print 'now i know my abc'
                    # a[n] = np.sign(np.dot(self.Mod_A[n , :].reshape(self.d, 1), q_prev))
                print 'hey jude'
                q_cur = self.normalize(self.mu(self.w, a, self.Mod_A, 0.1))
                if i > 10 and np.linalg.norm(q_prev - q_cur) <= max_err:
                    print 'dont make me cry'
                    break
                print 'sing a sad song'
                i += 1
            self.V[:, j] = q_cur
            print 'and make it better'
            self.Mod_A -= np.outer(np.dot(self.Mod_A, self.V[:, j]), self.V[:, j].T)
            print 'somehting else'
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
