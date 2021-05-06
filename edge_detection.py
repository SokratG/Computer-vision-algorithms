import numpy as np
import cv2
from math import ceil


def diff_img(img1, img2):
    result = np.uint8(abs(img1 - img2))
    return result

def magnitude_img(img_x, img_y):
    # calculate magnitude gradient horiz. and vertical.
    gradient_magnitude = np.sqrt(np.square(img_x) + np.square(img_y))
    # calc angle(direction) of gradient
    theta = np.arctan2(img_y, img_x)
    # normalize from 0 to 255
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    return (gradient_magnitude, theta)

def custom_conv(image, kernel):
    # calc padding image
    img_h, img_w = image.shape
    flt_h, flt_w = kernel.shape

    assert(flt_h > 0 and flt_w > 0)

    p_h = ceil((flt_h - 1) / 2)
    p_w = ceil((flt_w - 1) / 2)

    # new image with padding
    padding = np.zeros((img_h + (2*p_h), img_w + (2*p_w)))

    # put image inside padding
    padding[p_h:padding.shape[0] - p_h, p_w:padding.shape[1] - p_w] = image
    
    result = np.zeros(image.shape)
    # calc convolve for image
    for r in range(img_h):
        for c in range(img_w):
            result[r, c] = np.sum(kernel * padding[r:r+flt_h, c:c+flt_w])

    return result


# ---------------------------------- Sobel edge detection ----------------------------------
def customSobel(img_str, show=True):
    img = cv2.imread(img_str, cv2.IMREAD_GRAYSCALE)
    # filter Sovel for vertical and horizontal edges
    Sobel_flt_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sobel_flt_horizontal =  np.flip(np.transpose(Sobel_flt_vertical), axis=0) 

   
    img_y = custom_conv(img, Sobel_flt_vertical)
    img_x = custom_conv(img, Sobel_flt_horizontal)

    # return magnitude and angle   
    return magnitude_img(img_x, img_y)



# ---------------------------------- Prewitt edge detection ----------------------------------
def customPrewitt(img_str, show=True):
    img = cv2.imread(img_str, cv2.IMREAD_GRAYSCALE)
    # filter Prewitt for vertical and horizontal edges
    Prewitt_flt_vertical = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    Prewitt_flt_horizontal =  np.flip(np.transpose(Prewitt_flt_vertical), axis=0) 
    
    img_y = custom_conv(img, Prewitt_flt_vertical)
    img_x = custom_conv(img, Prewitt_flt_horizontal)

    # return magnitude and angle
    return  magnitude_img(img_x, img_y)


# ---------------------------------- Roberts edge detection ----------------------------------
def customRoberts(img_str, show=True):
    img = cv2.imread(img_str, cv2.IMREAD_GRAYSCALE)
    # filter Roberts for vertical and horizontal edges
    Roberts_flt_y = np.array([[1, 0], [0, -1]])
    Roberts_flt_x =  np.array([[0, 1], [-1, 0]]) 
    
    img_y = custom_conv(img, Roberts_flt_y)
    img_x = custom_conv(img, Roberts_flt_x)

    # return magnitude and angle
    return magnitude_img(img_x, img_y) 



# ---------------------------------- Canny edge detection ----------------------------------

def NonMaxSupression(img, angles):
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.float32)
    theta_ = (angles * 180) / np.pi
    theta_[theta_ < 0] += 180
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            p, c = 255, 255
            # find direction intensity
            if ((0. <= theta_[y][x] < 22.5) or (157.5 <= theta_[y][x] <= 180.)):
                p = img[y][x+1]
                c = img[y][x-1]
            elif (22.5 <= theta_[y][x] < 67.5):
                p = img[y+1][x-1]
                c = img[y-1][x+1]
            elif (67.5 <= theta_[y][x] < 112.5):
                p = img[y+1][x]
                c = img[y-1][x]
            elif (112.5 <= theta_[y][x] < 157.5):
                p = img[y-1][x-1]
                c = img[y+1][x+1]
            #check current pixel on local maximum
            if (img[y][x] >= p and img[y][x] >= c):
                result[y][x] = img[y][x]
            else:
                result[y][x] = 0
    result *= 255.0 / result.max()
    return np.uint8(result)

def findAround(img, x, y, val):
    # 8-connected pixels
    return ((img[y][x+1] == val) or (img[y+1][x] == val) or (img[y-1][x] == val) or (img[y][x-1] == val) or 
            (img[y+1][x+1] == val) or (img[y-1][x-1] == val) or (img[y-1][x+1] == val) or (img[y+1][x-1] == val))


def customHysteresis(img, weak_pixel, strong_pixel=255):
    h, w = img.shape
    result = np.copy(img)
    for y in range(1, h-1):
        for x in range(1, w-1):
            if (img[y][x] == weak_pixel):
                if (findAround(img, x, y, strong_pixel)):
                    result[y][x] = strong_pixel
                else:
                    result[y][x] = 0
    return result


def customDoubleThreshold(img, lowT, highT, weak_pixel, strong_pixel=255):
    h, w = img.shape
    result = np.zeros(img.shape)

    strong_y, strong_x = np.where(img >= highT)
    weak_y, weak_x = np.where((img <= highT) & (img >= lowT))   
    
    result[strong_y, strong_x] = strong_pixel
    result[weak_y, weak_x] = weak_pixel

    return result


def customCanny(img, weak_pixel, strong_pixel=255, low_threshold=None, high_threshold=None):

    sigma = 2
    ksize = 2 * sigma + 1 
    img_blur = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    
    sobel_edge, sobel_angle = customSobel(img_blur)
    
    EdgeNonMax = NonMaxSupression(sobel_edge, sobel_angle)
    
    if not low_threshold:
        low_threshold = EdgeNonMax.max() * 0.2
    if not high_threshold:
        high_threshold = EdgeNonMax.max() * 0.7

    ThreshImage = customDoubleThreshold(EdgeNonMax, low_threshold, high_threshold, weak_pixel, strong_pixel)
    
    result = customHysteresis(ThreshImage, weak_pixel, strong_pixel)
    
    return result
    
    
# -------------------------- LoG(Laplacian of Gaussian) edge detection ----------------------------
def kernel_Laplacian_of_Gaussian(sigma):
    # https://homepages.inf.ed.ac.uk/rbf/HIPR2/log.htm
    # second derivity of gaussian kernel
    ksize = np.ceil(6*np.sqrt(2)*sigma) # rec. kernel size = ceil(6*sqrt(2)*σ)xceil(6*sqrt(2)*σ)
    size = int(ksize) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    exp_ = np.exp(-(x ** 2 + y ** 2) / (2. * (sigma ** 2)))
    coef_ = 1/(np.pi * sigma**4) * ((x**2 + y**2 - 2 * sigma**2) / sigma**2)
    kernel = coef_ * exp_
    return kernel


def LoG_edge(img, sigma=1):
    import smooth
    assert(len(img.shape) == 2) # grayscale

    kernel = kernel_Laplacian_of_Gaussian(sigma)
    
    res_img = cv2.filter2D(img, -1, kernel)

    return res_img

