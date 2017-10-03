from pylab import *
from pylab import plt
import pca
import test
import utils
import empca_s
import empca_3
import rpca
import sys
sys.path.append('/Users/shaniacht/anaconda/lib/python2.7/site-packages')
import cv2
import skeleton
from skimage.measure import ransac, LineModelND
from skimage import feature,color
from timer import *
import pickle

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
        empcaResult = rpca.rpca(immatrix, m,n,100, 1.e-6, dim)
        timer.toc()

    return empcaResult.V, empcaResult.S, empcaResult.mean_A

# plt.close('all')
# imlist = utils.getImagesFromDir('./CasperIm/emptyLot/*.jpg')
#
# m,n = (array(plt.imread(imlist[0]))).shape[0:2]
#
# #########
# # img_rgb = utils.straightenImage(cv2.imread('./CasperIm/emptyLot/image1.jpg'))
# # cv2.imwrite('onepark.jpeg', img_rgb)
# # cv2.imshow('dd',img_rgb)
# # img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# # template = cv2.imread('./chacha.png',0)
# # w, h = template.shape[::-1]
# # res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# # threshold = 0.44
# # loc = np.where( res >= threshold)
# # for pt in zip(*loc[::-1]):
# #     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
# # cv2.imshow('res.png',img_rgb)
# # cv2.waitKey(0)
# #########
#
# img = utils.straightenImage(plt.imread(imlist[0]))
#
#
# f, ax = plt.subplots(figsize=(10, 10))
#
# edges = feature.canny(color.rgb2gray(img), sigma=3)
# edge_pts = np.array(np.nonzero(edges), dtype=float).T
# edge_pts_xy = edge_pts[:, ::-1]
#
# for i in range(4):
#     model_robust, inliers = ransac(edge_pts_xy, LineModelND, min_samples=2,residual_threshold=10, max_trials=1000)
#     x = np.arange(800)
#     figure('dd')
#     plt.plot(x, model_robust.predict_y(x))
#
#     edge_pts_xy = edge_pts_xy[~inliers]
#
# print 'hhhhh'
# plt.imshow(edges)
# plt.show()
#
# cv2.waitKey(0)
#
# 1/0
#
#
#
# cv2.imshow('original', img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('ff333f1',gray)
# ret3, edges = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
# cv2.imshow('fff1',edges)
# # edges = cv2.Canny(edges, 50, 200, apertureSize=3)
# # cv2.imshow('ffff2',edges)
# # img_BGR = cv2.cvtColor(edges, cv2.CV_GRAY2BGR)
# lines = cv2.HoughLinesP(edges, 10, np.pi / 180, 100, 150, 10)
# blank = np.zeros(img.shape, dtype=np.uint8)
# for x1, y1, x2, y2 in lines[0]:
#     cv2.line(blank, (x1, y1), (x2, y2), (0, 150, 0), 1)
# cv2.imshow('ff3',blank)
#
# # ts_dst = np.array([[485, 475], [620, 475], [485, 510], [620, 510]]).astype(np.float)
# blank = np.zeros(img.shape, dtype=np.uint8)
# x = 485
# y = 470 + 105
# width = 135
# height = 35
# for i in range (0, 9):
#     cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 1)
#     height += int(i * 0.8)
#     y = y - height
# cv2.imshow('fererererre',img)
# cv2.waitKey(0)
#
# gray = cv2.blur(gray, (3, 3))
# _, edges = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow('ddd',edges)
# skel = skeleton.skeletonize(edges)
# # gray = cv2.cvtColor(skel, cv2.COLOR_BGR2GRAY)
# gray = cv2.blur(skel, (3, 3))
# # _, edges = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
# cv2.imshow('cha cha', gray)
# minLineLength = 100
# maxLineGap = 0.1
#
# print img.shape
# blank = np.zeros(img.shape, dtype=np.uint8)
# lines = cv2.HoughLinesP(skel, 1, np.pi / 180, 50, 50, 3)
# for x1, y1, x2, y2 in lines[0]:
#     cv2.line(blank, (x1, y1), (x2, y2), (0, 150, 0), 1)
#
# cv2.imshow('cccc',blank)
# cv2.waitKey(0)
# 1/0
########



plt.close('all')
imlist = utils.getImagesFromDir('./CasperIm/emptyLot/*.jpg')

m,n = (array(plt.imread(imlist[0]))).shape[0:2]

N = len(imlist)
k = 2
timer = Timer()
immatrix = utils.getDataMatrix(imlist)

# calculate PCA/EMPCA
V,S,immean = calculatePCA('rpca', immatrix, m, n, k)

pickle_out = open("pca_results.pickle","wb")
pickle.dump(V, pickle_out)
pickle.dump(S, pickle_out)
pickle.dump(immean, pickle_out)
pickle_out.close()

pickle_out = open("pca_results.pickle", "r")
V = pickle.load(pickle_out)
S = pickle.load(pickle_out)
immean = pickle.load(pickle_out)
pickle_out.close()

print 'Showing PCA with deviation'
test.PCAwithDeviation(V,S, 4, immean, m ,n , 10)
print 'Reconstructing given image'
orig, guess, err = test.reconstructImageFromPCAModel(immean, './CasperIm/emptyLot/image111.jpg', k ,m , n, V)
print 'Checking equality of origin with reconstruction'
print test.checkEqual(orig, guess)
print 'Done'

plt.show()