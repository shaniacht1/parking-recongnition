import time

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def tic(self):
        self.tstart = time.time()
        print 'Start time is %d' % self.tstart

    def toc(self):
        endtime = time.time()
        print 'End time is %d' % endtime
        return endtime - self.tstart