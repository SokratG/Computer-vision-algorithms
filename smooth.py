import numpy as np
import cv2
from math import ceil


def color_renormalize(img):
    re_norm = img.copy()
    re_norm *= 255.0 / re_norm.max()
    re_norm = np.uint8(re_norm)
    return re_norm


# ---------------------------------- convolutional operation ----------------------------------
def custom_conv(image, kernel):
    img_h, img_w = image.shape
    flt_h, flt_w = kernel.shape
    assert(flt_h > 0 and flt_w > 0)
    p_h = ceil((flt_h - 1) / 2)
    p_w = ceil((flt_w - 1) / 2)
    padding = np.zeros((img_h + (2*p_h), img_w + (2*p_w)))
    padding[p_h:padding.shape[0] - p_h, p_w:padding.shape[1] - p_w] = image
    result = np.zeros(image.shape)
    for r in range(img_h):
        for c in range(img_w):
            result[r, c] = np.sum(kernel * padding[r:r+flt_h, c:c+flt_w])
    return color_renormalize(result)

# ---------------------------------- Gaussian distribution ----------------------------------
def gauss2d_custom(x, y, sigma):
    return np.exp(-(x ** 2 + y ** 2) / (2. * (sigma ** 2))) / (2.*np.pi*sigma)

def gauss_custom(value, sigma):
    return np.exp(-(value ** 2) / (2. * sigma ** 2)) / np.sqrt(2.*np.pi*sigma)

def gaussian_kernel(ksize, sigma=10):
    size = int(ksize) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    kernel =  gauss2d_custom(x, y, sigma)
    return kernel


# ---------------------------------- Gaussian blur ----------------------------------
def customGaussBlur(img, ksize=None, sigma=10):
    if not ksize:
        ksize = int(2 * sigma + 1)
    # create gauss kernel
    gaussKernel = cv2.getGaussianKernel(ksize**2, sigma, cv2.CV_32F).reshape(ksize, ksize)
    # apply guass filter
    return custom_conv(img, gaussKernel)


# ---------------------------------- bilateral blur ----------------------------------
def customBilateralBlur(org_img, sigma_spatial=10, sigma_intensity=10):
    img = color_renormalize(org_img)
   
    win_w = int(2 * sigma_spatial + 1)
    win_h = win_w
    
    Wp = np.zeros(img.shape) # * 1e-8
    result_img = np.zeros(img.shape) # + 1e-8

    for x in range(-win_w, win_w + 1):
        for y in range(-win_h, win_h + 1):
            
            offset = np.roll(img, [x, y], axis=[0, 1])
           
            gauss_calc = gauss2d_custom(x, y, sigma_spatial)
           
            temp_weight = gauss_calc * gauss_custom((offset - img), sigma_intensity)

            result_img += offset * temp_weight
           
            Wp += temp_weight
    result_img /= Wp 
    return result_img



# ---------------------------------- non local mean blur ----------------------------------
def customNLmeans(org_img, H=10, patch_big=7, patch_small=3): # super very slow!
    img = color_renormalize(org_img) 
   
    pad = patch_big + patch_small

    img_ = np.pad(img, pad, mode="reflect")

    result_img = np.zeros((img.shape[0], img.shape[1]))
    h, w = img_.shape

    sigma = 0
    
    for y in range(pad, h-pad):
        for x in range(pad, w-pad):

            current_val = 0
            startY = y - patch_big
            endY = y + patch_big

            startX = x - patch_big
            endX = x + patch_big
        
            Wp, maxweight = 0, 0
            for ypix in range(startY, endY):
                for xpix in range(startX, endX):
                    
                    window1 = img_[y-patch_small:y+patch_small, x-patch_small:x+patch_small].copy()
                    window2 = img_[ypix-patch_small:ypix+patch_small, xpix-patch_small:xpix+patch_small].copy()
                    # sigma not used, use H instead
                    weight = np.exp(-(np.sum((window1-window2)**2) + 2*(sigma**2)) / (H**2))

 
                    if weight > maxweight:
                        maxweight = weight
                    
                    if (y == ypix) and (x == xpix):
                        weight = maxweight

                    Wp += weight
                    current_val += weight * img_[ypix,xpix]
          
            result_img[y-pad,x-pad] = current_val/Wp
    return result_img



# ---------------------------------- Rudin–Osher–Fatemi Total Variation denoising  ----------------------------------
# https://www-pequan.lip6.fr/~bereziat/cours/master/vision/papers/rudin92.pdf
def ROF_denose(img, U, TV_weight, tolerance=0.1, tau=0.125):
    # img - grayscale image, U - initial hypothesis about image, TV_weight - total variation weight coefficient of regularization,
    # tau - step size,   tolerance - converge condition,

    U_ = U
    h, w = img.shape
    # init
    error = 1
    Px, Py = np.zeros((h, w)), np.zeros((h, w))

    while(error > tolerance):

        # gradient
        dUx = np.roll(U_, -1, axis=1) - U_
        dUy = np.roll(U_, -1, axis=0) - U_

        # recalc dual variable
        Px_ = Px + (tau/TV_weight) * dUx
        Py_ = Py + (tau/TV_weight) * dUy
        Norm_ = np.maximum(1, np.sqrt(Px_**2 + Py_**2))

        Px = Px_/Norm_
        Py = Py_/Norm_

        # recalc the primal variable
        RPx = np.roll(Px, 1, axis=1)
        RPy = np.roll(Py, 1, axis=0)

        # calc divergence of the dual field
        Div_ = (Px - RPx) + (Py - RPy)
        Uprev = U_

        # recalc new hypothesis value(primal variable)
        U_ = img + TV_weight * Div_ 

        error = np.linalg.norm(U_ - Uprev) / np.sqrt(h*w)

    # return denoising image and texture residual
    return U_, img - U_