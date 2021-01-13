import numpy as np
import scipy
import utilCV

# ---------------------------------- Camera class ----------------------------------
class Camera(object):
    def __init__(self, P):
        # P -camera matrix
        self.P = P
        self.K = None # calibration matrix(intrinsic)
        self.R = None # rotation matrix (extrinsic)
        self.t = None # translation matrix(extrinsic)
        self.c = None # center camera(intrinsic)
    
    def project(self, Xproj):
        # projection points from Xporj and normalize
        x = np.dot(self.P, Xproj)
        for i in range(3):
            x[i] /= x[2]
        return x
    
    def rotation_matrix(self, rvec):
        # make a rotation matrix around axis vector 'rvec' in 3-D space
        R = np.eye(4)
        A = np.array([[0, -rvec[2], rvec[1]], [rvec[2], 0, -rvec[0]], [-rvec[1], rvec[0], 0]])
        R[:3, :3] = scipy.linalg.expm(A)
        return R

    def get_cam_param(self):
        # RQ decompose camera matrix(parameters) in P = K[R|t]
        R, Q = scipy.linalg.rq(self.P[:, :3]) # K, R

        # make positive elements on diag K
        T = np.diag(np.sign(np.diag(R)))
        if np.linalg.det(T) < 0:
            T[1, 1] = np.abs(T[1, 1])
        
        self.K = np.dot(R, T)
        self.R = np.dot(T, Q)
        self.t = np.dot(np.linalg.inv(self.K), self.P[:, 3])

        return self.K, self.R, self.t

    def get_center(self):
        if self.c is not None:
            return self.c
        else:
            self.get_cam_param()
            self.c = -np.dot(self.R.transpose(), self.t)
            return self.c


def cam_calibration_(size, calib_size=(2555, 2586), resol=(2592, 1936)):
    h, w = size
    calib_h, calib_w = calib_size
    res_h, res_w = resol
    fx = calib_w * w / res_h
    fy = calib_h * h / res_w
    K = np.diag([fx, fy, 1])
    K[0][2] = 0.5 * w
    K[1][2] = 0.5 * h
    return K


# ---------------------------------- compute camera matrix ----------------------------------
def compute_3D_P(pts, pts3D): # compute from 3D points
    # x - points of image, X - 3D image points
    size_ = pts.shape[1]
    assert(size_ == pts3D.shape[1])

    M_ = np.zeros((3*size_, 12 + size_))
    for i in range(size_):
        M_[3*i, 0:4] = pts3D[:, i]
        M_[3*i+1, 4:8] = pts3D[:, i]
        M_[3*i+2, 8:12] = pts3D[:, i]
        M_[3*i:3*i+3, i+12] = -pts[:, i]

    _, _, V = np.linalg.svd(M_)

    # return first 12 element in the last eigenvector - camera matrix
    return V[-1, :12].reshape((3, 4))


def computer_E_P(E): # compute from essential matrix
    # computer second camera matrix from essential matrix on condition if camera 1 matrix P = [I|0]

    # check rang E matrix
    U, _, V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V

    E_ = np.dot(U, np.dot(np.diag([1, 1, 0]), V)) # I
    Z_ = utilCV.skew_sym([0, 0, -1])
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # 4 - solution for camera matrix
    P2 = [np.vstack((np.dot(U, np.dot(W, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W, V)).T, -U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:, 2])).T,
          np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:, 2])).T]

    return P2



