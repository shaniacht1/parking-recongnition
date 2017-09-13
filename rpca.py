from __future__ import division
import numpy as np
from scipy import stats

class rpca():
    def __init__(self, A, m, n, max_iter, max_err, k):
        self.N = A.shape[0]
        self.d = A.shape[1]
        self.mean_A = A.mean(axis=0)
        self.A = A - self.mean_A
        self.Mod_A = self.A.copy()
        self.V = np.zeros((self.d, k), dtype=np.float)  # matrix of size 2000000xk
        self.w = self.norms_vector(self.Mod_A).reshape(1, 50)
        self.rpca(k, max_iter, max_err)

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def norms_vector(self, V):
        return np.apply_along_axis(np.linalg.norm, 1, V)

    def normalizec(self, V):
        return np.apply_along_axis(self.normalize, 0, V)

    def normalizer(self, V):
        return np.apply_along_axis(self.normalize, 1, V)

    def mu(self, w, a, data, percent):
        times = np.multiply(a*w,data.T).T
        return stats.trim_mean(times, percent)

    def rpca(self, k, max_iter, max_err):
        a = np.zeros((1, self.N), dtype=np.float)
        j = 0
        while j < k:
            self.Mod_A = self.normalizer(self.Mod_A)
            print 'Calculating vector #', j
            i = 0
            q_cur = self.normalize(np.random.randint(0, 256, (1, self.d)).astype(np.float))
            while i < max_iter:
                print 'Iteration #', i, ' for vector #', j
                q_prev = q_cur
                for n in range(0, self.N - 1):
                    temp1 = self.Mod_A[n , :].reshape(self.d, 1)
                    temp2 = np.dot(q_prev,temp1)
                    a[0,n] = np.sign(temp2)
                q_cur = self.normalize(self.mu(self.w, a, self.Mod_A, 0.1))
                if i > 10 and np.linalg.norm(q_prev - q_cur) <= max_err:
                    break
                i += 1
            self.V[:, j] = q_cur
            self.Mod_A -= np.outer(np.dot(self.Mod_A, self.V[:, j]), self.V[:, j].T)
            j += 1

        alpha = np.dot(self.V.T, self.A.T)
        temp = np.cov(alpha)
        print 'temp type:',type(temp[0,0])
        self.S, eigVe = np.linalg.eigh(np.dot(0.02, temp))
        print 'temp fffff:', type(self.S[0]), type(eigVe[0,0])
        self.V = np.dot(self.V, eigVe).T
        print 'temp vvv:', type(self.V[0,0])
