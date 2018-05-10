from pylab import *
from pylab import plt
import pca
import visualizationUtils
import utils
import empca
import rpca
import sys
sys.path.append('../python2.7/site-packages')
from timer import *
import pickle

def calculatePCA(type, immatrix, m, n, dim):
    timer = Timer()
    if type is 'pca':
        timer.tic()
        V, S, immean = pca.pca(immatrix, m, n)
        timer.toc()
        return V, S, immean
    if type is 'empca':
        timer.tic()
        empcaResult = empca.empca(immatrix, 100, 1.e-6, dim, m, n)
        timer.toc()
    if type is 'rpca':
        timer.tic()
        empcaResult = rpca.rpca(immatrix, m,n,100, 1.e-6, dim)
        timer.toc()

    return empcaResult.V, empcaResult.S, empcaResult.mean_A

def createPCAPickle(src_dir, dst_file, src1, src2, src3, src4, dst1, dst2, dst3, dst4, pca_type = 'rpca'):
    imlist = utils.getImagesFromDir(src_dir+'/*.jpg')

    N = len(imlist)
    print 'number of photos is', N
    k = 20
    m,n = (array(plt.imread(imlist[0]))).shape[0:2]
    print 'm,n:', m,n

    immatrix = utils.getDataMatrix(imlist, src1, src2, src3, src4, dst1, dst2, dst3, dst4)

    # calculate PCA/EMPCA
    V,S,immean = calculatePCA(pca_type, immatrix, m, n, k)

    pickle_out = open(dst_file, "wb")
    pickle.dump(V, pickle_out)
    pickle.dump(S, pickle_out)
    pickle.dump(immean, pickle_out)
    pickle_out.close()

def showCreatedPCA(src_file, test_img_path, k, src1, src2, src3, src4, dst1, dst2, dst3, dst4):
    pickle_out = open(src_file, "r")
    V = pickle.load(pickle_out)
    S = pickle.load(pickle_out)
    immean = pickle.load(pickle_out)
    pickle_out.close()

    print '## Showing PCA results ##'
    testIm = utils.straightenImage(plt.imread(test_img_path), src1, src2, src3, src4, dst1, dst2, dst3, dst4).astype(np.float)
    print 'Showing PCA with deviation'
    m,n = testIm.shape[0:2]
    visualizationUtils.PCAwithDeviation(V, S, 4, immean, m, n, 10)
    print 'Reconstructing given image'
    orig, guess, err = visualizationUtils.reconstructImageFromPCAModel(immean, testIm, k, m, n, V)
    print 'Checking equality of origin with reconstruction'
    print visualizationUtils.checkEqual(orig, guess)
    print 'Done'

    plt.show()