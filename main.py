#from PIL import Image
#from numpy import *
from pylab import *
from pylab import plt
import pca
#import glob
import test
import utils
import empca_s
import empca_3

from timer import *

def calculatePCA(type, immatrix, m, n, dim):
    if type is 'pca':
        timer.tic()
        V, S, immean = pca.pca(immatrix, m, n)
        timer.toc()
        return V, S, immean
    if type is 'empca_s':
        timer.tic()
        empcaResult = empca_s.empca(immatrix, m, n, 100, 1.e-6, dim)
        timer.toc()
        return empcaResult.V, empcaResult.S, empcaResult.mean_A
    if type is 'empca_3':
        timer.tic()
        empcaResult = empca_3.empca(immatrix, 100, 1.e-6, dim, m, n)
        timer.toc()

        return empcaResult.V, empcaResult.S, empcaResult.mean_A

plt.close('all')
imlist = utils.getImagesFromDir('./CasperIm/emptyLot/*.jpg')
print imlist
print 'newwww'
m,n = (array(plt.imread(imlist[0]))).shape[0:2]
print 'size: ',m,n
N = len(imlist)
k = 30
timer = Timer()
immatrix = utils.getDataMatrix(imlist)

# calculate PCA/EMPCA
V,S,immean = calculatePCA('empca_3', immatrix, m, n, k)
# figure('immean')
# imshow(immean.astype(np.uint8).reshape(m, n, -1))
print 'ggg'
print S
print 'Showing PCA with deviation'
test.PCAwithDeviation(V,S, 4, immean, m ,n , 1)
print 'Reconstructing given image'
orig, guess, err = test.reconstructImageFromPCAModel(immean, './CasperIm/emptyLot/image47.jpg', k ,m , n, V)
print 'Checking equality of origin with reconstruction'
print test.checkEqual(orig, guess)
print 'Done'

#empca
#need to find eigenvalues

plt.show()