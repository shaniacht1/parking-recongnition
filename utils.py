from PIL import Image
#from numpy import *
from pylab import *
from pylab import plt
import pca
import glob
sys.path.append('/Users/shaniacht/anaconda/lib/python2.7/site-packages')
import cv2
from skimage import feature, color

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

def recDetection(img, blank):
    print 'fff',blank.shape
    print 'ffff', img.shape
    gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    #gray = cv2.GaussianBlur(gray, (1, 3), 0)
    # recognize less, but less shtuyot (more accurTE)
    ret3, edges = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # recognize more, but more shtuyot
    #edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boundRect = []
    polygons = []
    for i, cont in enumerate(contours):
        # poly = cv2.approxPolyDP(cont, 3, True)
        # polygons.append(poly)
        x, y, w, h = cv2.boundingRect(cont)
        if w < 200 and w > 100 and h > 20 and h < 60:
            cv2.rectangle(img, (x,y),(x+w, y+h), (0, 255, 0), 2)
    cv2.drawContours(img, contours, -1, (0, 0, 0), 1)
    cv2.imshow("cha cha", img)
    cv2.waitKey(0)

def lineDetection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    #_, edges = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 100, 300, apertureSize=3)

    minLineLength = 100
    maxLineGap = 0.1

    print img.shape
    blank = np.zeros(img.shape, dtype=np.uint8)
    lines = cv2.HoughLinesP(edges, 30, np.pi / 180, 20, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(blank, (x1, y1), (x2, y2), (0, 150, 0), 1)

    '''
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 125)
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    '''
    # cv2.imshow("cha cha",blank)
    # cv2.waitKey(0)
    return blank

def parkingPercents(orig_img,x,y,h,w,thresh):

    gray = cv2.cvtColor(orig_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    img = gray[y:y + h, x:x + w]
    _, bin_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    plt.imshow(bin_img)
    count = cv2.countNonZero(bin_img)
    plt.show()
    return count