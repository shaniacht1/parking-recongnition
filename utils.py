from PIL import Image
#from numpy import *
from pylab import *
from pylab import plt
import pca
import glob

def getImagesFromDir(path):
    imlist = []
    for filename in glob.glob(path):
        imlist.append(filename)
    return imlist


def getDataMatrix(imlist):
    L = []
    for f in imlist[:]:
        L.append(plt.imread(f))
    L = np.asarray(L).astype(np.float)
    return L.reshape(L.shape[0], -1)