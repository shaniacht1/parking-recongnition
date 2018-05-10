import math
import glob
import pickle
import statistics
import cv2
from pylab import *
from pylab import plt
sys.path.append('../python2.7/site-packages')
from skimage.morphology import skeletonize
from skimage.measure import ransac, LineModelND
from skimage import feature
import numpy as np

from Line import Line

def getImagesFromDir(path):
    imlist = []
    for filename in glob.glob(path):
        imlist.append(filename)
    return imlist


def cropImagesInDir(src_path, dst_path, x1, x2, y1, y2):
    imlist = getImagesFromDir(src_path+'/*.jpg')
    L = []
    for f in imlist[:]:
        image = plt.imread(f)[x1:x2, y1:y2]
        dst = dst_path +'/' +f.split('/')[-1]
        print dst
        cv2.imwrite(dst, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def getDataMatrix(imlist, src1, src2, src3, src4, dst1, dst2, dst3, dst4):
    L = []
    for f in imlist[:]:
        L.append(straightenImage(plt.imread(f), src1, src2, src3, src4, dst1, dst2, dst3, dst4))
    L = np.asarray(L).astype(np.float)
    return L.reshape(L.shape[0], -1)

def straightenImage(
        im_src,
        src1 = [301, 347], src2 = [479, 359],
        src3 = [249, 370], src4 = [439, 384],
        dst1 = [485, 475], dst2 = [620, 475],
        dst3 = [485, 510], dst4 = [620, 510]):

    pts_src = np.array([src1, src2, src3, src4]).astype(np.float)
    pts_dst = np.array([dst1, dst2, dst3, dst4]).astype(np.float)

    h, mask = cv2.findHomography(pts_src, pts_dst)
    im_dst = cv2.warpPerspective(im_src, h, (im_src.shape[1], im_src.shape[0]))

    return im_dst

def inliers_around_point(skeleton, x, y, win_w, win_h):
    y = int(round(y))
    m, n = 740, 1024
    x1 = max(0, x-win_w)
    x2 = min(n-1, x+win_w)
    y1 = max(0, y-win_h)
    y2 = min(m-1, y+win_h)
    cur_window = skeleton[y1 : y2, x1 : x2]
    return count_nonzero(cur_window)

def crop_lines(model_robust, skeleton, x, y, win_w, win_h):
    i = win_w
    cropped = []
    suspicious_count = 0
    suspicious_inter = 0
    start_x = -1
    isLine = False

    # initialization
    inliers_count = inliers_around_point(skeleton, x[win_w], y[win_w], win_w, win_h)
    if inliers_count > win_w*2-5:
        isLine = True
        start_x = 0
    # find line crops:
    while i < 1024:
        inliers_count = inliers_around_point(skeleton, i, int(y[i]), win_w, win_h) > (win_w*2-5)
        if isLine:
            if inliers_count:
                suspicious_count = 0
                inter_count = inliers_around_point(skeleton, x[i], int(y[i]), 5, 20)
                if inter_count > 30:
                    suspicious_inter+=1
                    if suspicious_inter > 5:
                        suspicious_inter = 0
                        # remove short lines
                        if i - start_x > 50:
                            newLine = Line(start_x, i, y[start_x], y[i], model_robust)
                            cropped.append(Line(start_x, i, y[start_x], y[i], model_robust))
                        start_x = x[i] + 5
                        i += 5
                else:
                    suspicious_inter = 0
            else:
                suspicious_count+=1
                if suspicious_count >= win_w:
                    suspicious_count = 0
                    if x[i] - start_x > 100:
                        cropped.append(Line(start_x, i-win_w, y[start_x], y[i-win_w], model_robust))
                    isLine = False
        else:
            if inliers_count:
                suspicious_count += 1
                if suspicious_count >= win_w:
                    suspicious_count = 0
                    start_x = i-win_w
                    isLine = True
            else:
                suspicious_count = 0
        i+=1
    if isLine and start_x < x.shape[0]:
        cropped.append(Line(start_x, x.shape[0]-1-70, y[start_x], y[(x.shape[0]-1-70)], model_robust))
    return cropped

def lineDetection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = feature.canny(gray, sigma=3)
    blur_edges = cv2.blur((edges * 255).astype(np.uint8), (9, 9))
    _, bin_edges = cv2.threshold(blur_edges, 127, 255, cv2.THRESH_OTSU)
    bin_edges = bin_edges > 0
    skeleton = skeletonize(bin_edges.astype(np.uint8))
    edge_pts = np.array(np.nonzero(skeleton), dtype=float).T
    edge_pts_xy = edge_pts[:, ::-1]

    points_per_line = []
    hor_lines = []
    ver_lines = []

    # get all lines in array
    # model_robust.predict - might be useful to find intersections
    for i in range(25):
        model_robust, inliers = ransac(edge_pts_xy, LineModelND, min_samples=2, residual_threshold=10, max_trials=1000)
        points_per_line += [inliers]

        x1 = np.arange(img.shape[1])
        y1 = model_robust.predict_y(x1)
        y2 = np.arange(img.shape[0])
        x2 = model_robust.predict_x(y2)

        # this is a vertical line
        if np.abs(y1[0] - y1[-1]) > 730 and np.abs(x2[0] - x2[-1]) < 20:
            ver_lines.append(Line(x2[0], x2[-1], y2[0], y2[-1], model_robust))
        # this is a horizontal line
        if np.abs(y1[0] - y1[-1]) < 50 and np.abs(x2[0] - x2[-1]) > 1000:
            cur_lines = crop_lines(model_robust, skeleton, x1, y1, 10, 10)
            # figure('Line Markings')
            # plt.scatter([x.x1 for x in cur_lines], [x.y1 for x in cur_lines], c="g")
            # plt.scatter([x.x2 for x in cur_lines], [x.y2 for x in cur_lines], c="r")
            hor_lines += cur_lines
        edge_pts_xy = edge_pts_xy[~inliers]
    # plt.imshow(skeleton)
    # plt.show()
    return hor_lines, ver_lines

def areNeigbours(line1, line2):
    if not line1 or not line2:
        return False
    x11 = line1.x1
    x12 = line1.x2
    x21 = line2.x1
    x22 = line2.x2
    return np.abs(x11-x21) < 30 and np.abs(x12-x22) < 30 and np.abs(line1.y1 - line2.y1) < 65 and np.abs(line1.y1 - line2.y1) > 15

def special_x_sort(line):
    x = line.x1
    return int(math.ceil(x / 100.0)) * 100

def find_parkings(img):
    plt.imshow(img)
    plt.show()
    hor_lines, ver_lines = lineDetection(img)
    sorted_hor_lines = sorted(sorted(hor_lines, key=lambda x: x.y1), key=special_x_sort)
    parkings = []
    i=0
    is_next_neig = False
    prev_line = sorted_hor_lines[0]
    prev_points = [(prev_line.x1, int(prev_line.y1)),(prev_line.x2,int(prev_line.y2))]
    while i<len(sorted_hor_lines):
        cur_line = sorted_hor_lines[i]
        cur_x_start = cur_line.x1
        cur_x_end = cur_line.x2
        if i < len(sorted_hor_lines) - 1:
            next_line = sorted_hor_lines[i+1]
            is_next_neig = areNeigbours(cur_line, next_line)
        is_prev_neig = areNeigbours(cur_line, prev_line)
        if is_prev_neig:
            #  is_prev_neig &&  is_next_neig => line in the middle of row
            if is_next_neig:
                cur_x1 = statistics.median([prev_points[0][0],cur_x_start,next_line.x1])
                cur_x2 = statistics.median([prev_points[1][0],cur_x_end,next_line.x2])
            # is_prev_neig && !is_next_neig => line last of row
            else:
                cur_x1 = cur_x_start if np.abs(prev_points[0][0]-cur_x_start) < 5  else prev_points[0][0]
                cur_x2 = cur_x_end if np.abs(prev_points[1][0]-cur_x_start) < 5  else prev_points[1][0]
        else:
            # !is_prev_neig &&  is_next_neig => line first of row
            if is_next_neig:
                cur_x1 = cur_x_start
                cur_x2 = cur_x_end
            # !is_prev_neig && !is_next_neig => not a parking!
            else:
                cur_line = None
                cur_x1 = -1000
                cur_x2 = -1000
        if cur_line:
            cur_points = [(cur_x1, int(cur_line.y1)), (cur_x2, int(cur_line.y2))]
            cur_points_orig = [(cur_x_start, int(cur_line.y1)), (cur_x_end, int(cur_line.y2))]
        else:
            cur_points = [(cur_x1, -1000), (cur_x2, -1000)]
            cur_points_orig = [(cur_x_start, -1000), (cur_x_end, -1000)]
        if is_prev_neig:
            parkings.append(cur_points + prev_points)
        prev_line = cur_line
        prev_points = cur_points_orig
        i+=1


    pickle_out = open("parking_array.pickle","wb")
    pickle.dump(parkings, pickle_out)
    pickle_out.close()

    return parkings