from PIL import Image
#from numpy import *
from pylab import *
from pylab import plt
import pca
import glob
sys.path.append('/Users/shaniacht/anaconda/lib/python2.7/site-packages')
import cv2

def getImagesFromDir(path):
    imlist = []
    for filename in glob.glob(path):
        imlist.append(filename)
    return imlist


def getDataMatrix(imlist):
    L = []
    for f in imlist[:]:
        L.append(straightenImage(plt.imread(f)))
    L = np.asarray(L).astype(np.float)
    return L.reshape(L.shape[0], -1)

def straightenImage(im_src):
    #im_src = cv2.imread(imlist[43])
    pts_src = np.array([[301, 347], [479, 359], [249, 370], [439, 384]]).astype(np.float)
    # pts_dst = np.array([[346, 323], [518, 323], [346, 377],[518, 377]]).astype(np.float)
    pts_dst = np.array([[485, 475], [620, 475], [485, 510], [620, 510]]).astype(np.float)

    h, status = cv2.findHomography(pts_src, pts_dst)

    im_dst = cv2.warpPerspective(im_src, h, (im_src.shape[1], im_src.shape[0]))
    return im_dst

    # cv2.imshow("Source Image", im_src)
    # cv2.imshow("Destination Image", im_dst)
    # cv2.waitKey(0)

    # finish test homogrpahy