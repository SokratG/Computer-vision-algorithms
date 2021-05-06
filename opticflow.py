import numpy as np
import cv2
from math import ceil
import utilCV

# ---------------------------------- Optical flow Horn-Schunck method ----------------------------------
# http://www.cs.cmu.edu/~16385/s17/Slides/14.3_OF__HornSchunck.pdf
# http://www.sci.utah.edu/~gerig/CS6320-S2013/Materials/CS6320-CV-S2012-OpticalFlow-I.pdf
def Horn_Schunck_opt_flow(img0, img1, lambda_=0.4, T=7):
  
    img_h, img_w = img0.shape
    # recommend init even arrays ?
    # add some padding with size = 1
    u = np.zeros((img_h + 2, img_w + 2)) 
    v = np.zeros((img_h + 2, img_w + 2))

    img0_norm = img0 / 255.0
    img1_norm = img1 / 255.0
    Ix, Iy = np.gradient(img0_norm) 
    It = cv2.subtract(img0_norm, img1_norm) 

    n = 1
    while n <= T:
        for y in range(img_h):
            for x in range(img_w):
                u_mean = (u[y, x - 1] + u[y, x + 1] + u[y + 1, x] + u[y - 1, x]) / 4 # average 4-connectivity
                v_mean = (v[y, x - 1] + v[y, x + 1] + v[y + 1, x] + v[y - 1, x]) / 4
                alpha = (Ix[y, x] * u_mean + Iy[y, x] * v_mean + It[y, x]) / ((lambda_**2)  + (Ix[y, x]**2) + (Iy[y, x]**2))
                u[y, x] = u_mean - alpha * Ix[y, x]
                v[y, x] = v_mean - alpha * Iy[y, x]
        n += 1

    # back to normal size 
    res_u = u[1:u.shape[0] - 1, 1:u.shape[1] - 1] # try re-normalize?
    res_v = v[1:v.shape[0] - 1, 1:v.shape[1] - 1]
    

    return (res_u , res_v)



# ---------------------------------- Optical flow Lucas-Kanade method ----------------------------------
# http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf
# http://www.cs.cmu.edu/~16385/s15/lectures/Lecture21.pdf
def Lucas_Kanade_opt_flow(img0, img1, T=4, ksize=5, threshold=0.00001, sigma=1.5):
    from scipy.ndimage import filters

    img_h, img_w = img0.shape
    # recommend init even arrays ?
    # add some padding with size = 1
    u = np.zeros((img_h, img_w)) 
    v = np.zeros((img_h, img_w))
    Ix, Iy = np.zeros(img0.shape), np.zeros(img0.shape)

    #Ix, Iy = np.gradient(img0)
    filters.gaussian_filter(img0, (sigma, sigma), (0,1), Ix)
    filters.gaussian_filter(img1, (sigma, sigma), (1,0), Iy)   

    It = cv2.subtract(img0, img1) 
    
    n = 1
    k = np.int32(np.floor(ksize / 2.0))
    k_w = np.int32(np.ceil(ksize / 2.0))
    while n <= T:
        for y in range(k, img_h-k):
            for x in range(k, img_w-k):
                A = np.vstack((Ix[y-k:y+k_w, x-k:x+k_w].flatten(), Iy[y-k:y+k_w, x-k:x+k_w].flatten())).T            
                b = -(It[y-k:y+k_w, x-k:x+k_w].flatten())
                A_t = np.matmul(A.T, A)
                l1, l2 = np.linalg.eigvals(A_t)

                # eiganvals should not be a small
                if (l1 < threshold or l2 < threshold):
                    continue

                b_t = np.matmul(A.T, b)
                # A'*A - should be invertible
                try:
                    solve_x = np.matmul(np.linalg.inv(A_t), b_t)
                except np.linalg.LinAlgError:
                    continue

                u[y, x] = solve_x[0]
                v[y, x] = solve_x[1]
        n += 1
    

    return (u , v)