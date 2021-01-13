import numpy as np
from math import ceil
import cv2


def customThreshold(img, limitval=127, isInv=False):
    h, w = img.shape
    assert(w > 0 and h > 0)
    bin_img = np.copy(img)
    val = 255
    if (isInv):
        val = 0
    for y in range(h):
        for x in range(w):
            pix = img[y, x]          
            if (pix > limitval):
                bin_img[y][x] = val
            else:
                bin_img[y][x] = 255-val
    return bin_img



#--------------------------------------------------------
def prepareMorph(bin_img, h, w, kernel_h, kernel_w):

    assert(kernel_h > 1 and kernel_w > 1)
   
    pad_x, pad_y = ceil((kernel_h-1)/2), ceil((kernel_w-1)/2)
    
    _img = np.zeros((h + (2*pad_y), w + (2*pad_x)))
   
    temp_img = np.zeros(_img.shape)
    
    temp_img[pad_y:temp_img.shape[0]-pad_y, pad_x:temp_img.shape[1]-pad_x] = bin_img

    return (_img, temp_img, pad_y, pad_x)


# ---------------------------------- Erode ----------------------------------
def customErode(bin_img, kernel): # only for rect primitive!
    h, w = bin_img.shape
    kernel_h, kernel_w = kernel.shape

    erode_img, temp_img, pad_y, pad_x = prepareMorph(bin_img, h, w, kernel_h, kernel_w)

    kernel_sum = kernel.sum()
    cx, cy = pad_x, pad_y
    for y in range(h):
        for x in range(w):
            sub_window = temp_img[y:y+kernel_h, x:x+kernel_w]
            sub_sum = (sub_window*kernel).sum() / 255
            if (kernel_sum == sub_sum): # change this line for using other primitive
                erode_img[y+cy][x+cx] = 255

    return erode_img[:w][:h]


# ---------------------------------- Dilate ----------------------------------
def customDilate(bin_img, kernel): # only for rect primitive!
    h, w = bin_img.shape
    kernel_h, kernel_w = kernel.shape

    dilate_img, temp_img, pad_y, pad_x = prepareMorph(bin_img, h, w, kernel_h, kernel_w)

    cx, cy = pad_x, pad_y
    for y in range(h):
        for x in range(w):
            sub_window = temp_img[y:y+kernel_h, x:x+kernel_w]
            sub_sum = (sub_window*kernel).sum()
            if (sub_sum != 0): # change this line for using other primitive
                dilate_img[y+cy][x+cx] = 255

    return dilate_img[:w][:h]

# ---------------------------------- Opening ----------------------------------
def customOpening(bin_img, kernel): # only for rect primitive!
    open_img = customErode(bin_img, kernel)
    open_img = customDilate(open_img, kernel)

    return open_img


# ---------------------------------- Closing ----------------------------------
def customClosing(bin_img, kernel): # only for rect primitive!
    close_img = customDilate(bin_img, kernel)
    close_img = customErode(close_img, kernel)
    
    return close_img