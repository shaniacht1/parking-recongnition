from __future__ import division
import pickle
import skimage
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float,img_as_ubyte
from skimage import io
import matplotlib.pyplot as plt
import argparse
import numpy as np
import cv2
import utils
from PIL import Image, ImageDraw



def calcPercents(img,x1,y1,x2,y2,x3,y3,x4,y4):
    y = min(y1, y3)
    x = min(x1, x3)
    w=np.abs(x4-x1)
    h=np.abs(y1-y4)
    park = img[y:y + h, x:x + w]
    count = cv2.countNonZero(park)
    return  (count/(h*w))*100

def showPercents(rectangles,org_img,error_img,tresh):
    org_img = (img_as_ubyte(org_img))
    draw_rec=np.zeros(org_img.shape,dtype=(np.uint8))
    for rec in rectangles:
        percents=calcPercents(tresh,rec[0][0],rec[0][1],rec[1][0],rec[1][1],rec[2][0],rec[2][1],rec[3][0],rec[3][1])
        parking=decide(rec[0],rec[1],rec[2],rec[3],percents)
        color=(39,215,51)
        if parking==0:
            color=(251,18,47)
        if parking==1:
            color=(255,153,0)
        cv2.rectangle(draw_rec,(rec[0][0],rec[0][1]),(rec[3][0],rec[3][1]),(255,0,0),3)
        reversed,h=utils.reverseImage(error_img)
        original = np.array([((int((rec[3][0]+rec[0][0])/2),int((rec[0][1] + rec[3][1])/2)),(int((rec[3][0]+rec[0][0] + rec[1][0] + rec[2][0])/4),int((rec[0][1] + rec[3][1] + rec[2][1]+ rec[1][1])/4)))], dtype=np.float32)
        converted = cv2.perspectiveTransform(original, h)
        cv2.circle(org_img,(converted[0][0][0],converted[0][0][1]), 13,color, -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(org_img,str(int(percents))+'%',(converted[0][1][0],converted[0][1][1]), font, 0.4,(255,255,255),1,cv2.LINE_AA)

    # reverse_rectangles, m = utils.reverseImage(draw_rec)
    plt.imshow(org_img)
    plt.show()

    pass

def decide(p1,p2,p3,p4,percents):
    if percents>68:
        return 0
    if percents>50 and percents <=67 :
        return 1
    else:
        return 2

def mean_spx_calc(segments,image,numSegments):
    means=[]
    new_map=np.zeros(image.shape,dtype=np.float)

    for i in range(len(np.unique(segments))):
        means.append(np.mean(image[segments==i]))
        new_map[segments==i]=means[i]
    return means,new_map

def slic():
    err_img=r".\error.jpeg"
    image = (plt.imread(err_img))[:,:,0:3]
    org_img=(plt.imread(r"./sourceData/image.jpg"))[:,:,0:3]
    img_grey = img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    numSegments=4000
    segments = slic(img_grey, n_segments = numSegments,compactness=0.1, sigma = 5)
    #fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    #ax = fig.add_subplot(1, 1, 1)
    #ax.imshow(mark_boundaries(img_grey, segments))
    #plt.axis("off")
    means,new_map=mean_spx_calc(segments,img_grey,numSegments)
    ret,tresh = cv2.threshold(img_as_ubyte(new_map),39,255,cv2.THRESH_BINARY)

    pickle_out=open("parking_array.pickle","r")
    recs=pickle.load(pickle_out)
    pickle_out.close()
    showPercents(recs,img_as_float(org_img),img_as_float(image),tresh)


