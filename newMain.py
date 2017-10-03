from pylab import *
from pylab import plt
import numpy as np
import sys
from skimage.morphology import skeletonize
import sys
sys.path.append('/Users/shaniacht/anaconda/lib/python2.7/site-packages')
import cv2
from skimage.measure import ransac, LineModelND
from skimage import feature
import utils


#########
# img_rgb = utils.straightenImage(cv2.imread('./CasperIm/emptyLot/image1.jpg'))
# cv2.imwrite('onepark.jpeg', img_rgb)
# cv2.imshow('dd',img_rgb)
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# template = cv2.imread('./chacha.png',0)
# w, h = template.shape[::-1]
# res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
# threshold = 0.44
# loc = np.where( res >= threshold)
# for pt in zip(*loc[::-1]):
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
# cv2.imshow('res.png',img_rgb)
# cv2.waitKey(0)
#########
img = (plt.imread('nanana.jpeg'))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)


print utils.parkingPercents(img,485, 475, 35, 135,250)
plt.show()
cv2.waitKey(0)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = feature.canny(gray, sigma=3)
blur_edges = cv2.blur((edges * 255).astype(np.uint8), (9, 9))
_, bin_edges = cv2.threshold(blur_edges, 127, 255, cv2.THRESH_OTSU)
bin_edges = bin_edges > 0
skeleton = skeletonize(bin_edges.astype(np.uint8))

edge_pts = np.array(np.nonzero(skeleton), dtype=float).T
edge_pts_xy = edge_pts[:, ::-1]
points_per_line = []
for i in range(21):
    model_robust, inliers = ransac(edge_pts_xy, LineModelND, min_samples=2, residual_threshold=10, max_trials=1000)
    points_per_line += [inliers]
    x = np.arange(1000)
    figure('dd')
    plt.plot(x, model_robust.predict_y(x))
    edge_pts_xy = edge_pts_xy[~inliers]

plt.imshow(skeleton)


#############################################################

M = img.shape[0]
N = img.shape[1]

bw = np.zeros((M,N),dtype=np.bool)

h=35
w=135

# location
x=485
y=475

bw[y:y+h,x:x+w]=1

plt.figure(1)
plt.clf()
plt.subplot(241)
plt.imshow(bw,interpolation="None")

d1=cv2.distanceTransform(bw.astype(np.uint8),cv2.cv.CV_DIST_L2,3)
d2=cv2.distanceTransform((1-bw).astype(np.uint8),cv2.cv.CV_DIST_L2,3)

d=d1-d2
d=np.abs(d)

plt.subplot(242)
plt.imshow(d1)
plt.subplot(243)
plt.imshow(d2)
plt.subplot(244)
plt.imshow(d)

plt.subplot(245)

pattern = d1[y:y+h,x:x+w]
multi_pattern=np.tile(pattern,[5,1])
plt.imshow(multi_pattern)

plt.show()
cv2.waitKey(0)

