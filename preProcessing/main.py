from pylab import *
import utils
import sys
sys.path.append('../python2.7/site-packages')
import createPCA

src_dir = './sourceData'
dst_dir = './sourceDataCropped'
pca_result_file = 'pca.pickle'

#homogrpahy - to be used in the createPCA methofdfdfdf

# If crop is necessary
# crop_vals = [530, 680, 100, 650]
# utils.cropImagesInDir(src_dir, dst_dir,crop_vals[0], crop_vals[1], crop_vals[2], crop_vals[3])
# src_dir = dst_dir

# Create the PCA pickle file, to be user later in the postProcessing stage
createPCA.createPCAPickle(src_dir, pca_result_file, src1, src2, src3, src4, dst1, dst2, dst3, dst4)
print 'pickle created'

# createPCA.showCreatedPCA(pca_result_file, './GermanParkingCropped/16.jpg', 5, src1, src2, src3, src4, dst1, dst2, dst3, dst4)


