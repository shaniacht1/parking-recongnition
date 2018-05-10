from pylab import *
from pylab import plt
import numpy as np
import sys
sys.path.append('../python2.7/site-packages')
import cv2
from skimage.measure import ransac, LineModelND
from skimage import feature
import utils
from PIL import Image, ImageEnhance
import slic


image = Image.open('./sourceData/image.jpg')
plt.imshow(image)
plt.show()
contrast = ImageEnhance.Sharpness(image)
new_img = np.array(contrast.enhance(2))
plt.imshow(new_img)
plt.show()

src1 = [85,137]
src2 = [69,43]
src3 = [484,113]
src4 = [374,33]
dst1 = [250,175]
dst2 = [790,175]
dst3 = [250,490]
dst4 = [790,490]

# Use the same homogrphy as for the PCA
img = utils.straightenImage(new_img,src1, src2, src3, src4, dst1, dst2, dst3, dst4)

plt.imshow(img)
plt.show()
parking_array = utils.find_parkings(img)

rec1 = np.zeros(img.shape, dtype=(np.uint8))
for rec in parking_array:
    cv2.rectangle(rec1, (rec[0][0], rec[0][1]),(rec[3][0], rec[3][1]), (255,0,0), 3)
img[rec1 > 0] = 0

slic.slic()

