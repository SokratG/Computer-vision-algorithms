import numpy as np
from scipy.ndimage import filters
import cv2

# ---------------------------------- Harris-Stephens corner detector  ----------------------------------
# https://en.wikipedia.org/wiki/Harris_Corner_Detector
def corner_detector(img, sigma=3, min_dist=10, threshold=0.1):
    # img - given image, min_dist - minimal distance between corner, 
    # threshold for cornerns

    imgX, imgY = np.zeros(img.shape), np.zeros(img.shape)

    # derivation
    filters.gaussian_filter(img, (sigma, sigma), (0,1), imgX)
    filters.gaussian_filter(img, (sigma, sigma), (1,0), imgY)

    # Harris-Stephens matrix
    Wxx = filters.gaussian_filter(imgX*imgX, sigma)
    Wxy = filters.gaussian_filter(imgX*imgY, sigma)
    Wyy = filters.gaussian_filter(imgY*imgY, sigma)

	# Compute the response of the detector at each pixel
    # Noble's 1998 - "DESCRIPTIONS OF IMAGE SURFACES"
    Wdet = Wxx * Wyy - Wxy**2
    Wtrace = Wxx + Wyy
    W = np.divide(Wdet, Wtrace, where=Wtrace!=0) # out=np.zeros_like(Wdet)
    
    # apply threshold for Harris corners
    corner_thresh = W.max() * threshold
    Harris_thresh = (W > corner_thresh) * 1

    # get coordinates and values of Harris-Stephens corners
    coords_Harris = np.array(Harris_thresh.nonzero()).transpose()
    values_Harris = np.array([W[c_[0], c_[1]] for c_ in coords_Harris])

    # sort corners by value on descending order
    sort_idx = np.argsort(values_Harris, kind='quicksort')[::-1]
    
    # save data about point location
    pts_location = np.zeros(W.shape)
    pts_location[min_dist:-min_dist, min_dist:-min_dist] = 1
    
    # store best points corners
    flt_coords = []
    for i in sort_idx:
        if (pts_location[coords_Harris[i, 0], coords_Harris[i, 1]] == 1):
            flt_coords.append(coords_Harris[i])
            x_step_min = coords_Harris[i,0] - min_dist
            x_step_max = coords_Harris[i,0] + min_dist
            y_step_min = coords_Harris[i,1] - min_dist
            y_step_max = coords_Harris[i,1] + min_dist
            pts_location[x_step_min:x_step_max, y_step_min:y_step_max] = 0


    # return size of corners points and list of corner points
    return flt_coords


# ---------------------------------- Descriptor features  ----------------------------------
def calc_desciptors(img, Harris_score_corners, px_width):
    # img - given grayscale image, Harris_score_corners - list of Harris-Stephens corner, 
    # px_width - patch size = 2*px_width+1 for descriptions(px_width < minimal distance Harris corner)
    descriptors = []
    for c in Harris_score_corners:
        # store the linearization patch
        patch_ = img[c[0]-px_width:c[0]+px_width+1, c[1]-px_width:c[1]+px_width+1].flatten()
        descriptors.append(patch_)
    
    # returb size of list descriptors and list of descriptors
    return descriptors

# ---------------------------------- Matches features  ----------------------------------
def match_features(desc1, desc2, threshold=0.5):
    # desc1 - list with descriptors of image1, desc2 - list with descriptors of image2,
    # threshold - for filtering normalized cross correlation

    # get patch size
    patch_size = len(desc1[0])
   
    # find best pairwise distance
    dist = -np.ones((len(desc1), len(desc2))) # change sign, because bigger value is better
    for i in range(len(desc1)):
        d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
        for j in range(len(desc2)):
            d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
            # calc normalized cross correlation
            ncc_val = np.sum(d1 * d2) / (patch_size - 1)
            if (ncc_val > threshold):
                dist[i, j] = ncc_val

    match_desc = np.argsort(-dist)[:, 0]

    # return match between two descpriptors
    return match_desc
    


def find_best_matches(desc1, desc2, threshold=0.5):

    match1_2 = match_features(desc1, desc2, threshold)
    match2_1 = match_features(desc2, desc1, threshold)

    best_idx = np.where(match1_2 >= 0)[0]

    # exclude non symmetric matches
    for i in best_idx:
        if match2_1[match1_2[i]] != i:
            match1_2[i] = -1

    return match1_2

# ---------------------------------- Matches and SIFT Features ----------------------------------
def features_and_match_OPENCV(img1, img2, ratio=0.9):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good_pts1, good_pts2 = [], []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good_pts2.append(kp2[m.trainIdx].pt)
            good_pts1.append(kp1[m.queryIdx].pt)      
            
    pts1 = np.int32(good_pts1)
    pts2 = np.int32(good_pts2)        

    return pts1, pts2

# ---------------------------------- HOG descriptors (Dense SIFT) ----------------------------------
def hog_descs(img, visualisize=False, px_per_cell=(10, 10), cells_per_block=(5, 5), orientation=8):
    # return HOG descriptors
    from skimage import feature
    if (visualisize):
        hogdesc, hog_image = feature.hog(img, orientations=orientation, 
                                    pixels_per_cell=px_per_cell, cells_per_block=cells_per_block, 
                                    block_norm='L2-Hys', visualize=visualisize)
        return hogdesc, hog_image
    else:
        hogdesc = feature.hog(img, orientations=orientation, 
                                    pixels_per_cell=px_per_cell, cells_per_block=cells_per_block, 
                                    block_norm='L2-Hys', visualize=False)
        return hogdesc

    return None
    
# ---------------------------------- Compute Hessian Image ----------------------------------
def Hessian(img, sigma=0):

    imgX, imgY = np.zeros(img.shape, dtype=np.float32), np.zeros(img.shape, dtype=np.float32)
    # imgYX and imgXY are equal in this case?
    imgXX, imgYY = np.zeros(img.shape, dtype=np.float32), np.zeros(img.shape, dtype=np.float32)
    imgXY, imgYX = np.zeros(img.shape, dtype=np.float32), np.zeros(img.shape, dtype=np.float32) 

    if sigma > 0: # use gaussian kernel
        # first derivation
        filters.gaussian_filter(img, (sigma, sigma), (0,1), imgX)
        filters.gaussian_filter(img, (sigma, sigma), (1,0), imgY)      

        # second derivation
        filters.gaussian_filter(imgX, (sigma, sigma), (0,1), imgXX)
        filters.gaussian_filter(imgX, (sigma, sigma), (1,0), imgXY)
        filters.gaussian_filter(imgY, (sigma, sigma), (0,1), imgYX)
        filters.gaussian_filter(imgY, (sigma, sigma), (1,0), imgYY)
    else:
        # first derivation
        imgX, imgY = np.gradient(img)
        # second derivation
        imgXX, imgXY = np.gradient(imgX)
        imgYX, imgYY = np.gradient(imgY)
    

    return (imgXX, imgYY, imgXY)
