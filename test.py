from PIL import Image
#from numpy import *
from pylab import *
from pylab import plt
import pca
import glob
import utils

def checkEqual(orig, new):
    return np.allclose(orig, new)

def showReconstruction(orig, guess,error,m,n):
    figure('Original image vs. reconstructed image')
    subplot(1, 2, 1)
    imshow(orig.astype(np.uint8))
    subplot(1, 2, 2)
    imshow(guess.astype(np.uint8).reshape(m, n, -1))
    figure('Reconstruction error visualization')
    imshow(error.astype(np.uint8).reshape(m, n, -1))

def reconstructImageFromPCAModel(immean, path, dim, m, n, V):
    testIm = utils.straightenImage(plt.imread(path)).astype(np.float)
    V = V[:dim]

    alpha = np.dot(V, (testIm.ravel() - immean))
    guessX = immean + np.dot(np.transpose(V), alpha)
    error = testIm.ravel() - guessX
    showReconstruction(testIm, guessX, error, m,n)
    return testIm.ravel(), guessX, error

def PCAwithDeviation(V, S, dim, immean, m , n, sigma):
    figure('PCA vectors with mean' + `sigma`)
    for i in range(0, dim):
        #subplot(4, np.ceil(dim/4).astype(np.int)+1, i + 1)
        subplot(2, 2, i + 1)
        if(S is not -1):
            v = (V[i] * sigma * np.sqrt(S[i]) + immean).reshape(m, n, -1)
        else:
            v = V[i].reshape(m,n,-1)
        imshow(v.astype(np.uint8))