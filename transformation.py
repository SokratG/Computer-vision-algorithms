import numpy as np
import cv2
import scipy.ndimage 
import utilCV

# ---------------------------------- Homography Matrix ----------------------------------
# http://vhosts.eecs.umich.edu/vision//teaching/EECS442_2011/lectures/discussion2.pdf
def find_homography(src_pts, proj_pts):
    # src_pts - points on some source image, proj_pts - projection points from source image
    # find homography matrix use direct linear transformation(DLT)
    assert(src_pts.shape == proj_pts.shape)

    # calc mean and standard deviation for source and project image
    mean_src = np.mean(src_pts[:2], axis=1)
    mean_proj = np.mean(proj_pts[:2], axis=1)
    max_std_src = np.max(np.std(src_pts[:2], axis=1)) + 1e-8 # avoid divide by 0
    max_std_proj = np.max(np.std(proj_pts[:2], axis=1)) + 1e-8 # avoid divide by 0


    # find matrix A for pairwise corresponding points in A*h=0 for DLT method
    # normalize points with mean and std for stable calculating
    T1 = np.diag([1/max_std_src, 1/max_std_src, 1])
    T2 = np.diag([1/max_std_proj, 1/max_std_proj, 1])
    T1[0][2] = -mean_src[0] / max_std_src 
    T1[1][2] = -mean_src[1] / max_std_src 
    T2[0][2] = -mean_proj[0] / max_std_proj 
    T2[1][2] = -mean_proj[1] / max_std_proj 
    A_src = np.dot(T1, src_pts)
    A_proj = np.dot(T2, proj_pts)
    
    size_neighbor = A_src.shape[1]
    A = np.zeros((2*size_neighbor))
    for i in range(size_neighbor):
        A[2*i] = [-A_src[0][i], -A_src[1][i], -1, 0, 0, 0,
                  A_proj[0][i]*A_src[0][i],A_proj[0][i]*A_src[1][i], A_proj[0][i]]
        A[2*i+1] = [0, 0, 0, -A_src[0][i], -A_src[1][i], -1,
                    A_proj[1][i]*A_src[0][i], A_proj[1][i]*A_src[1][i], A_proj[1][i]]
        
    # find Homography matrix H using SVD decomposition
    _, _, V = np.linalg.svd(A) # need only unitary singular matrix Vh(less std error)
    H_ = np.dot(np.linalg.inv(T2), np.dot(V[8].reshape(3, 3), T1))
    H = H_ / H_[2, 2] # normalize

    return H



def find_homography_affine(src_pts, proj_pts):
    assert(src_pts.shape == proj_pts.shape)
    mean_src = np.mean(src_pts[:2], axis=1)
    mean_proj = np.mean(proj_pts[:2], axis=1)
    max_std = np.max(np.std(src_pts[:2], axis=1)) + 1e-8  # avoid divide by 0
    
    # ------
    T1 = np.diag([1/max_std, 1/max_std, 1])
    T2 = T1.copy() #np.diag([1/max_std_proj, 1/max_std_proj, 1])
    T1[0][2] = -mean_src[0] / max_std 
    T1[1][2] = -mean_src[1] / max_std 
    T2[0][2] = -mean_proj[0] / max_std 
    T2[1][2] = -mean_proj[1] / max_std 
    A_src = np.dot(T1, src_pts)
    A_proj = np.dot(T2, proj_pts)
    # ------

    # mean of points equal 0, therefore parallel translation vector is a 0
    A = np.concatenate((A_src[:2], A_proj[:2]), axis=0)
    _, _, V = np.linalg.svd(A.transpose())
    tmp_ = V[:2].transpose()
    B = tmp_[:2]
    C = tmp_[2:4]
    
    tmp_ = np.concatenate((np.dot(C, np.linalg.pinv(B)), np.zeros((2, 1))), axis=1)
    H_ = np.vstack((tmp_,[0, 0, 1]))
    H_ = np.dot(np.linalg.inv(T2), np.dot(H_, T1))
    H = H_ / H_[2, 2] # normalize

    return H



# ---------------------------------- Affine warp on triangle patches ----------------------------------
def partial_warp_affine(src_img, proj_img, src_pts, proj_pts, triangles):

    res_img = proj_img.copy()
    temp_img = np.zeros(res_img.shape, np.uint8)
    
    for tri_idx in triangles:
        # calc homography matrix
        H = find_homography_affine(proj_pts[:, tri_idx], src_pts[:,tri_idx])
        
        # check color image
        if len(src_img.shape) == 3:
            for ch in range(src_img.shape[2]): # ch - channel
                temp_img[:, :, ch] = scipy.ndimage.affine_transform(src_img[:, :, ch], H[:2, :2], (H[0,2], H[1, 2]), res_img.shape[:2])
        else:
            temp_img = scipy.ndimage.affine_transform(src_img, H[:2, :2], (H[0,2], H[1, 2]), res_img.shape[:2])
        alpha = utilCV.alpha_for_triangle(proj_pts[:, tri_idx], res_img.shape[0], res_img.shape[1])
        
        res_img[alpha > 0] = temp_img[alpha > 0]

    return res_img


# ---------------------------------- RANSAC ----------------------------------
def find_Homography_RANSAC(img1, img2, ratio=0.9, ransacReprojThresh=0.5):
    # https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
    MIN_MATCH_COUNT = 25

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good_pts = []
    for m,n in matches:
        if m.distance < ratio*n.distance:
            good_pts.append(m)

    if len(good_pts) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_pts ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_pts ]).reshape(-1,1,2)
    else:
        return None
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThresh)

    return H


# ---------------------------------- Fundamental Matrix ----------------------------------
def find_fundamental_matrix(pts1, pts2):
    # pts1, pts2 - points from image1 and image2 respectively 
    # find fundamental matrix using normalize 8-points algorithm.

    size_ = pts1.shape[1]
    assert(pts2.shape[1] == size_)

    # construct matrix for constrain equation
    A = np.zeros((size_, 9))
    for i in range(size_):
        A[i] = [pts1[0, i]*pts2[0, i],pts1[0, i]*pts2[1, i], pts1[0, i]*pts2[2, i],
                pts1[1, i]*pts2[0, i],pts1[1, i]*pts2[1, i], pts1[1, i]*pts2[2, i],
                pts1[2, i]*pts2[0, i],pts1[2, i]*pts2[1, i], pts1[2, i]*pts2[2, i]]

    # calc linear least square
    _, _, V = np.linalg.svd(A)
    F_ = V[-1].reshape(3, 3)

    # constrain matrix F_ and make rang matrix F_ eq. 2(last singular value = 0)
    # minimization of mean squared error 
    U, S, V = np.linalg.svd(F_)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    # return fundametnal coordinate matrix
    return F #/ F[2, 2]

# ---------------------------------- Epipole ----------------------------------
def find_epipole(F):
    # compute epipole from fundamental matrix F

    # Fx = 0
    _, _, V = np.linalg.svd(F)
    e = V[-1] # last element

    # return right epipole(for left must transpose result)
    return e / e[2]

# ---------------------------------- Triangulate points ----------------------------------
def trianglulate_2pt_(pt1, pt2, cameraP1, cameraP2):
    # compute triangulation 2 points. P1, P2 - camera matrix
    size_ = 6
    A = np.zeros((size_, size_))
    A[:3, :4] = cameraP1
    A[3:, :4] = cameraP2
    A[:3, 4] = -pt1
    A[:3, 5] = -pt2

    _, _, V = np.linalg.svd(A)
    tri_pts = V[-1, :4] # last 4 elem eigenvec 
    # normalize
    return tri_pts / tri_pts[3]


def triangulate(pts1, pts2, cameraP1, cameraP2):
    size_ = pts1.shape[1]
    assert(size_ == pts2.shape[1])

    Tr = np.array([trianglulate_2pt_(pts1[:, i], pts2[:, i], cameraP1, cameraP2) for i in range(size_)]).transpose()

    return Tr


# ---------------------------------- Fundamental Matrix RANSAC ----------------------------------
def find_fundamental_normalized(pts1, pts2):
    # Computes the fundamental matrix from corresponding points using the normalized 8 point algorithm
       
    size_ = pts1.shape[1]
    assert(size_ == pts2.shape[1])

    # normalize image coordinates
    pts1 = pts1 / pts1[2]
    mean_1 = np.mean(pts1[:2], axis=1)
    S1 = np.sqrt(2) / np.std(pts1[:2]) # 2 ?
    T1 = np.array([[S1, 0,-S1 * mean_1[0]], [0, S1 ,-S1 * mean_1[1]], [0,0,1]])
    pts1_ = np.dot(T1, pts1)
    
    pts2 = pts2 / pts2[2]
    mean_2 = np.mean(pts2[:2], axis=1)
    S2 = np.sqrt(2) / np.std(pts2[:2]) # 2 ?
    T2 = np.array([[S2, 0,-S2 * mean_2[0]], [0, S2, -S2 * mean_2[1]], [0,0,1]])
    pts2_ = np.dot(T2, pts2)

    # compute F with the normalized coordinates
    F_ = compute_fundamental(pts1_, pts2_)

    F = np.dot(T1.transpose(), np.dot(F_, T2))
    # return normalized fundamental matrix
    return F / F[2,2]

# ---------------------------------- Fundamental Matrix using OPENCV ----------------------------------
def find_F_RANSAC(pts1, pts2, maxiter=5000, threshold=1e-6):
    # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
    F, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=3., confidence=threshold, maxIters=maxiter)
    return F, inliers



