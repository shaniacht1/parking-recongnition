from PIL import Image
#from numpy import *
from pylab import *
from pylab import plt
import pca
import glob
sys.path.append('/Users/shaniacht/anaconda/lib/python2.7/site-packages')
import cv2
from skimage.measure import ransac, LineModelND
from skimage.morphology import skeletonize
from skimage.measure import ransac, LineModelND
from skimage import feature
import statistics
import pickle
import math
import Line

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

    h, mask = cv2.findHomography(pts_src, pts_dst)
    im_dst = cv2.warpPerspective(im_src, h, (im_src.shape[1], im_src.shape[0]))
    return im_dst

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

def inliers_around_point(skeleton, x, y, win_w, win_h):
    # print 'this is y: ',round(y)
    # print 'this is win_h: ', win_h
    y = int(round(y))
    # print len(skeleton)
    m, n = 740, 1024
    # print 'this is m,n: ', m,n
    x1 = max(0, x-win_w)
    x2 = min(n-1, x+win_w)
    y1 = max(0, y-win_h)
    y2 = min(m-1, y+win_h)
    # print x1 , x2, y1, y2
    # window_area = np.abs((x1-x2)*(y1-y2))
    cur_window = skeleton[y1 : y2, x1 : x2]
    # plt.imshow(cur_window)
    # plt.show()
    # print 'nonzero in window:', count_nonzero(cur_window)
    return count_nonzero(cur_window)

def crop_lines(model_robust, skeleton, x, y, win_w, win_h):
    print 'crop lines started'
    print 'x shape is: ', x.shape[0]
    print 'y shape is: ', y.shape[0]
    i = win_w
    cropped = []
    suspicious_count = 0
    suspicious_inter = 0
    start_x = -1
    end_x = -1
    isLine = False
    #initialization
    inliers_count = inliers_around_point(skeleton, x[win_w], y[win_w], win_w, win_h)
    if inliers_count > win_w*2-5:
        print 'line started', 0
        isLine = True
        start_x = 0
    # find line crops:
    # for i in range(win_w, x.shape[0]):
    while i < x.shape[0]:
        # print i
        inliers_count = inliers_around_point(skeleton, i, int(y[i]), win_w, win_h) > (win_w*2-5)
        if isLine:
            if inliers_count:
                suspicious_count = 0
                inter_count = inliers_around_point(skeleton, x[i], int(y[i]), 5, 20)
                if inter_count > 30:
                    suspicious_inter+=1
                    if suspicious_inter > 5:
                        suspicious_inter = 0
                        print 'found intersection', i, i, y[start_x]
                        # remove short lines
                        if i - start_x > 50:
                            cropped.append((model_robust, start_x, i, y[start_x], y[i]))
                        start_x = x[i] + 5
                        i += 5
                else:
                    suspicious_inter = 0
            else:
                suspicious_count+=1
                if suspicious_count >= win_w:
                    print 'line finished', i, i, y[start_x]
                    suspicious_count = 0
                    if x[i] - start_x > 100:
                        cropped.append((model_robust, start_x, i-win_w, y[start_x], y[i-win_w]))
                    isLine = False
        else:
            if inliers_count:
                suspicious_count += 1
                if suspicious_count >= win_w:
                    print 'line started', i, x[i], y[start_x]
                    suspicious_count = 0
                    start_x = i-win_w
                    isLine = True
            else:
                suspicious_count = 0
        i+=1
    if isLine:
        cropped.append((model_robust, start_x, x.shape-1, y[start_x], y[x.shape-1]))
    return cropped

def findIntersections(hor_lines, ver_lines):
    return hor_lines

def lineDetection(img):
    print 'img size: ', img.shape
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
    for i in range(21):
        model_robust, inliers = ransac(edge_pts_xy, LineModelND, min_samples=2, residual_threshold=10, max_trials=1000)
        points_per_line += [inliers]

        x1 = np.arange(1024)
        y1 = model_robust.predict_y(x1)
        y2 = np.arange(768)
        x2 = model_robust.predict_x(y2)
        # this is a vertical line
        if np.abs(y1[0] - y1[-1]) > 730 and np.abs(x2[0] - x2[-1]) < 20:
            ver_lines.append((model_robust, np.abs(x2[0] - x2[-1])/2))
        # this is a horizontal line
        if np.abs(y1[0] - y1[-1]) < 50 and np.abs(x2[0] - x2[-1]) > 1000:
            # hor_lines += [(model_robust, x1, y1)]
            cur_lines = crop_lines(model_robust, skeleton, x1, y1, 10, 10)
            figure('dd')
            plt.scatter([x[1] for x in cur_lines], [x[3] for x in cur_lines], c="g")
            plt.scatter([x[2] for x in cur_lines], [x[3] for x in cur_lines], c="r")
            hor_lines += cur_lines
        edge_pts_xy = edge_pts_xy[~inliers]
    plt.imshow(skeleton)
    return hor_lines, ver_lines

def y_value(line):
    return line[3]

def x_start_value(line):
    return line[1]

def x_end_value(line):
    return line[2]

def areNeigbours(line1, line2):
    if not line1 or not line2:
        return False
    x11 = x_start_value(line1)
    x12 = x_end_value(line1)
    x21 =  x_start_value(line2)
    x22 = x_end_value(line2)
    return np.abs(x11-x21) < 30 and np.abs(x12-x22) and np.abs(line1[3] - line2[3]) < 50

def special_x_sort(line):
    x = x_start_value(line)
    return int(math.ceil(x / 100.0)) * 100

def find_parkings(img):
    hor_lines, ver_lines = lineDetection(img)
    sorted_hor_lines = sorted(sorted(hor_lines, key=y_value), key=special_x_sort)
    parkings = []
    i=0
    prev_line = sorted_hor_lines[0]
    prev_model = prev_line[0]
    y_values = prev_model.predict_y([x_start_value(prev_line), x_end_value(prev_line)])
    prev_points = [(x_start_value(prev_line), int(y_values[0])),(x_end_value(prev_line),int(y_values[1]))]
    print sorted_hor_lines
    while i<len(sorted_hor_lines):
        cur_line = sorted_hor_lines[i]
        cur_x_start = x_start_value(cur_line)
        cur_x_end = x_end_value(cur_line)
        if i < len(sorted_hor_lines) - 1:
            next_line = sorted_hor_lines[i+1]
            is_next_neig = areNeigbours(cur_line, next_line)
        is_prev_neig = areNeigbours(cur_line, prev_line)
        #  is_prev_neig &&  is_next_neig => line in the middle of row
        #  is_prev_neig && !is_next_neig => line last of row
        # !is_prev_neig &&  is_next_neig => line first of row
        # !is_prev_neig && !is_next_neig => not a parking!
        if is_prev_neig:
            if is_next_neig:
                cur_x1 = statistics.median([prev_points[0][0],cur_x_start,x_start_value(next_line)])
                cur_x2 = statistics.median([prev_points[1][0],cur_x_end,x_end_value(next_line)])
            else:
                cur_x1 = cur_x_start if np.abs(prev_points[0][0]-cur_x_start) < 5  else prev_points[0][0]
                cur_x2 = cur_x_end if np.abs(prev_points[1][0]-cur_x_start) < 5  else prev_points[1][0]
        else:
            if is_next_neig:
                cur_x1 = cur_x_start
                cur_x2 = cur_x_end
            else:
                cur_line = None
                cur_x1 = -1000
                cur_x2 = -1000
        if cur_line:
            cur_points = [(cur_x1, int(cur_line[3])), (cur_x2, int(cur_line[4]))]
        else:
            cur_points = [(cur_x1, -1000), (cur_x2, -1000)]
        if is_prev_neig:
            parkings.append(cur_points + prev_points)
        prev_line = cur_line
        prev_points = cur_points
        i+=1

    print len(sorted_hor_lines)
    print parkings

    pickle_out = open("parking_array.pickle","wb")
    pickle.dump(parkings, pickle_out)
    pickle_out.close()

    return parkings