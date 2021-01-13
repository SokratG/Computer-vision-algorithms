import numpy as np
import scipy
import cv2
import pylab
import utilCV
import os

# ---------------------------------- Image registration ----------------------------------
def calc_rigid_transform(refpts, pts):
    # calc rotation angle, scale coef and translation coef
    assert(len(refpts) == len(pts))

    A = np.array([ [pts[0], -pts[1], 1, 0],
                   [pts[1], -pts[0], 0, 1],
                   [pts[2], -pts[3], 1, 0],
                   [pts[3], -pts[2], 0, 1],
                   [pts[4], -pts[5], 1, 0],
                   [pts[5], -pts[4], 0, 1]])
    y = np.array([refpts[0],
                  refpts[1],
                  refpts[2],
                  refpts[3],
                  refpts[4],
                  refpts[5]])

    # method least squares with minimize L1-norm
    a, b, tx, ty = scipy.linalg.lstsq(A, y)[0] # np : scipy ?
    # a, b - parametric?
    R = np.array([[a, -b], [b, a]]) # rotate with scale coef

    return R, tx, ty

def rigid_alignment(img_pts, path, pad=20, isSave=False):
    # img_pts - images points, path - file path for save result
    # isometric combine images and save them
    
    refpts = list(img_pts.values())[0]
    res_imgs = []
    for im in img_pts:
        pts = img_pts[im]

        R_, tx, ty = calc_rigid_transform(refpts, pts)
        R = np.array([[R_[1][1], R_[1][0]],[R_[0][1], R_[0][0]]])
    
        s_img = cv2.imread(os.path.join(path, im), cv2.IMREAD_UNCHANGED)
        img_ = np.zeros(s_img.shape, np.uint8)

        for i in range(len(s_img.shape)):
            img_[:, :, i] = scipy.ndimage.affine_transform(s_img[:, :, i], np.linalg.inv(R), offset=[-ty, -tx])

        # cutting border and save image
        h, w = img_.shape[:2]
        border = int((h + w) / pad) # padding
        if isSave:
            cv2.imwrite(os.path.join(path, 'aligned/' + im), img_[border:h-border, border:w-border])
        res_imgs.append(img_[border:h-border, border:w-border])

    # return alignment images on given points
    return res_imgs




# ---------------------------------- Stereo Image ----------------------------------
# uniform filter
def plane_sweep_ncc(img1, img2, start, steps, width):
    # find map disparity using norm Cross-correlation
    h, w = img1.shape
    mean_1 = np.zeros((h, w))
    mean_2 = np.zeros((h, w))
    s_ = np.zeros((h, w))
    s_1 = np.zeros((h, w))
    s_2 = np.zeros((h, w))

    # array with depth plane
    depth_map = np.zeros((h, w, steps))
    # compute mean on block    
    scipy.ndimage.filters.uniform_filter(img1, width, mean_1)
    scipy.ndimage.filters.uniform_filter(img2, width, mean_2)

    # norm images
    norm_1 = img1 - mean_1
    norm_2 = img2 - mean_2

    # use different disparity
    for d in range(steps):
        scipy.ndimage.filters.uniform_filter(np.roll(norm_1, -d-start)*norm_2, width, s_)
        scipy.ndimage.filters.uniform_filter(np.roll(norm_1, -d-start)*np.roll(norm_1, -d-start), width, s_1)
        scipy.ndimage.filters.uniform_filter(norm_2*norm_2, width, s_2)
    
        # store estimate ncc
        depth_map[:, :, d] = s_ / np.sqrt(s_1 * s_2)

    # return best depth for every pixel
    return np.argmax(depth_map, axis=2)

# gauss filter
def plane_sweep_ncc_gauss(img1, img2, start, steps, width):
    # find map disparity using norm Cross-correlation
    DEFAULT_GAUSS = 0
    h, w = img1.shape
    mean_1 = np.zeros((h, w))
    mean_2 = np.zeros((h, w))
    s_ = np.zeros((h, w))
    s_1 = np.zeros((h, w))
    s_2 = np.zeros((h, w))

    # array with depth plane
    depth_map = np.zeros((h, w, steps))
    # compute mean on block    
    scipy.ndimage.filters.gaussian_filter(img1, width, DEFAULT_GAUSS, mean_1)
    scipy.ndimage.filters.gaussian_filter(img2, width, DEFAULT_GAUSS, mean_2)

    # norm images
    norm_1 = img1 - mean_1
    norm_2 = img2 - mean_2

    # use different disparity
    for d in range(steps):
        scipy.ndimage.filters.gaussian_filter(np.roll(norm_1, -d-start)*norm_2, width, DEFAULT_GAUSS, s_)
        scipy.ndimage.filters.gaussian_filter(np.roll(norm_1, -d-start)*np.roll(norm_1, -d-start), width, DEFAULT_GAUSS, s_1)
        scipy.ndimage.filters.gaussian_filter(norm_2*norm_2, width, DEFAULT_GAUSS, s_2)
    
        # store estimate ncc
        depth_map[:, :, d] = s_ / np.sqrt(s_1 * s_2)

    # return best depth for every pixel
    return np.argmax(depth_map, axis=2)


# ---------------------------------- Cartoon animation ----------------------------------
# https://towardsdatascience.com/turn-photos-into-cartoons-using-python-bb1a9f578a7e

def cartoon_animate(img, ksize=5, bsize=9, C=3, sigma_space=20, sigma_color=20, K=9, criteria=None, maxiter=20, eps=1.0):
    # img - image, ksize = kernel smooth size, bsize - blocksize threshold size, C - constant subtracted from the calculated mean in adaptive threshold
    # sigma color - std color for bilateral filter, sigma space - std space for bilateral filter, K - number of colors parameter in kmeans algorithm
    # criteria - for kmeans algorithm, maxiter and eps - parameters of criteria 
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edge detect
    gblur_ = cv2.medianBlur(gimg, ksize)
    edges_ = cv2.adaptiveThreshold(gblur_, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bsize, C)

    # reduce black artefacts
    kernel_ = np.ones((3, 3), np.uint8)
    edges_ = cv2.morphologyEx(edges_, cv2.MORPH_CLOSE, kernel_, iterations=1, borderType=cv2.BORDER_REPLICATE)
    #edges = cv2.morphologyEx(edges_, cv2.MORPH_DILATE, kernel_, iterations=1, borderType=cv2.BORDER_REPLICATE)

    # color quantization
    kmdata = np.float32(img).reshape((-1, 3))
    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, maxiter, eps) # 20 -max iteration, 1.0 - epsilon
    attemps = 10
    # K - 
    _, bestlabel, centroids = cv2.kmeans(kmdata, K, None, criteria, attemps, cv2.KMEANS_RANDOM_CENTERS)
    centroids = np.uint8(centroids)
    img_quant = centroids[bestlabel.flatten()].reshape(img.shape)

    bimg = cv2.bilateralFilter(img_quant, ksize, sigma_color, sigma_space) # nlmean?

    result_cartoon = cv2.bitwise_and(bimg, bimg, mask=edges_)
    
    return result_cartoon


# ---------------------------------- TODO Panorama ----------------------------------


