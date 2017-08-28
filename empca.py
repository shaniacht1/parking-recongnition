import numpy as np

class empca():
    def __init__(self,Y, m, n, max_iter, max_err, k):
        self.mean_Y = Y.mean(axis=0)
        self.Y = Y - self.mean_Y
        self.C = np.random.randn(Y.shape[0], k)
        print 'tata'
        print self.C.shape
        self.V, self.S = self.empca(m, n, max_iter, max_err, k)
        print 'EMPCA init complete'

    def e_step(self):
        print 'EMPCA e_step started'
        Ct = self.C.T
        invCtC = np.linalg.inv(np.dot(Ct,self.C))
        CtY = np.dot(Ct, self.Y)
        self.X = np.dot(invCtC,CtY)
        print 'EMPCA e_step finished'


    def m_step(self):
        print 'EMPCA m_step started'
        Xt = self.X.T
        YXt = np.dot(self.Y, Xt)
        invXXt = np.linalg.inv(np.dot(self.X, Xt))
        self.C = np.dot(YXt, invXXt)
        print 'EMPCA m_step finished'


    def ls_error(self):
        print 'EMPCA ls_error started'
        proj = np.dot(self.C, self.X)
        return np.sum((self.Y - proj)**2.0)

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm ==0:
            return v
        return v/norm

    def empca(self, m, n, max_iter, max_err, k):
        last_error = np.Inf
        dif_error = np.Inf
        print 'EMPCA started'

        i=0
        while(i<max_iter and dif_error > max_err):
            self.e_step()
            self.m_step()
            cur_error = self.ls_error()
            dif_error = last_error - cur_error
            last_error = cur_error
            i += 1
            if i == max_iter:
                print 'max iter reached'

        print 'in empca'
        print np.apply_along_axis(self.normalize, 0,self.C).shape
        print self.C.shape
        return np.apply_along_axis(self.normalize, 0,self.C), (np.sqrt(np.sum(self.C ** 2, axis=0)))
