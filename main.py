from pylab import *
from pylab import plt
import pca
import test
import utils
import empca_s
import empca_3
import rpca

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
    if type is 'empca_3':
        timer.tic()
        empcaResult = empca_3.empca(immatrix, 100, 1.e-6, dim, m, n)
        timer.toc()
    if type is 'rpca':
        timer.tic()
        empcaResult = rpca.rpca(immatrix, m,n,100, 1.e-6, dim, None)
        timer.toc()

    return empcaResult.V, empcaResult.S, empcaResult.mean_A
print np.multiply(np.array([1, -1]),np.array([[1,2,3],[4,5,6]]).T).T
plt.close('all')
imlist = utils.getImagesFromDir('./CasperIm/emptyLot/*.jpg')
print imlist
print 'newwww'
m,n = (array(plt.imread(imlist[0]))).shape[0:2]
print 'size: ',m,n
N = len(imlist)
k = 5
timer = Timer()
immatrix = utils.getDataMatrix(imlist)

# calculate PCA/EMPCA
V,S,immean = calculatePCA('rpca', immatrix, m, n, k)
# figure('immean')
# imshow(immean.astype(np.uint8).reshape(m, n, -1))
print 'Showing PCA with deviation'
test.PCAwithDeviation(V,S, 4, immean, m ,n , 1)
test.PCAwithDeviation(V,S, 4, immean, m ,n , 100)
test.PCAwithDeviation(V,S, 4, immean, m ,n , 50)
test.PCAwithDeviation(V,S, 4, immean, m ,n , -100)
test.PCAwithDeviation(V,S, 4, immean, m ,n , -50)


print 'Reconstructing given image'
orig, guess, err = test.reconstructImageFromPCAModel(immean, './CasperIm/image991-31_Aug_2017_7_41.jpg', k ,m , n, V)
print 'Checking equality of origin with reconstruction'
print test.checkEqual(orig, guess)
print 'Done'

plt.show()