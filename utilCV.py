import os
from matplotlib import pyplot as plt
import pylab
from scipy.spatial import Delaunay
import numpy as np
from xml.dom import minidom
#from skimage.util import random_noise


# ---------------------------------- get all files in directory ----------------------------------
def get_dir_files(path, endstring='.*'):
    return [os.path.join(path, file) for file in os.listdir(path) if file.endswith(endstring)]

# ---------------------------------- invert colors in image ----------------------------------
def invert_color(img):
    img_ = 255 - img[:, :]
    return img_

# ---------------------------------- constrain intensity image in interval ----------------------------------
def interval_contrast(img, min=0, max=255):
    img_ = ((max-min) / 255) * img[:, :] + min
    return  np.uint8(img_)

def square_contrast(img):
    img_ = 255. * (img[:, :]/255.)**2
    return  np.uint8(img_)

# ---------------------------------- compute average images ----------------------------------
def compute_avgimgs(imgslist):
    size_ = imgslist.size
    if (size_ == 0):
        return np.array([], np.uint8)
    avr_img = imgslist[0]
    for img in imgslist[1:]:
        avr_img += img
    avr_img /= size
    return np.uint8(avr_img)


# ---------------------------------- show image matplotlib ----------------------------------
def show_img(img):
    plt.imshow(img, cmap='gray')
    plt.title("result image")
    plt.show()
    return

# ---------------------------------- show color image matplotlib ----------------------------------
def show_img_color(img):
    plt.imshow(img)
    plt.title("result image")
    plt.show()
    return

# ---------------------------------- show images matplotlib ----------------------------------
def show_compare(images, titles):
    if (len(images) > 6):
        return
    size = len(images)
    for i in range(size):
        plt.subplot(2, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap='gray')
    plt.show()
    return

# ---------------------------------- show images color matplotlib ----------------------------------
def show_compare_color(images, titles, h=1, w=1):
    if (len(images) > 6):
        return
    size = len(images)
    for i in range(size):
        plt.subplot(h, w, i + 1)
        plt.title(titles[i]) 
        plt.imshow(images[i])
    plt.axis('off')
    plt.show()
    return
    

# ---------------------------------- show images matplotlib ----------------------------------
def show_compare_r(images, titles, h=1, w=1):
    if (len(images) > 6):
        return
    size = len(images)
    for i in range(size):
        plt.subplot(h, w, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap='gray')
    plt.axis('off')
    plt.show()
    return

# ---------------------------------- color normalize and scale [0..255] ----------------------------------
def color_renormalize(img):
    re_norm = img.copy()
    re_norm *= 255.0 / re_norm.max()
    re_norm = np.uint8(re_norm)
    return re_norm

# ---------------------------------- add noise to image ----------------------------------
'''
def add_noise(img, var_=0.04):
    #mean = 0
    #var = 200
    #sigma = var**0.5
    #gauss = np.random.normal(mean, sigma, (img.shape))
    #noise = gauss.reshape(img.shape)
    #noise_img = img + noise
    noise_img = random_noise(img, mode='gaussian', var=var_**2)
    return noise_img
'''

def add_noise_r(img, sigma=10):
    h, w = img.shape
    noise_img = np.float32(img.copy())
    noise_img += sigma * np.random.standard_normal((h, w))
    return noise_img

# ---------------------------------- concat to images ----------------------------------
def concat_imgs(img1, img2):

    # recalc rows of images
    if (img1.shape[0] < img2.shape[0]):
        img1 = np.concatenate((img1, np.zeros(img2.shape[0] - img1.shape[0], img1.shape[1])), axis=0)
    elif (img1.shape[0] > img2.shape[0]):
        img2 = np.concatenate((img2, np.zeros(img1.shape[0] - img2.shape[0], img2.shape[1])), axis=0)

    # return concatenation image
    return np.concatenate((img1, img2), axis=1)

# ---------------------------------- alpha blend ----------------------------------
def alpha_blend(img1,img2,alpha):
    """ Blend two images with weights as in alpha. """
    return (1 - alpha)*img1 + alpha * img2  

# ---------------------------------- draw labels ----------------------------------
def draw_labels(img, labels):
    pylab.imshow(img)
    pylab.contour(labels, [-0.5, 0.5])
    pylab.contourf(labels, [-1, -0.5], colors='b', alpha=0.3)
    pylab.contourf(labels, [0.5, 1], colors='r', alpha=0.3)
    pylab.axis('off')

    return

# ---------------------------------- transform to homogeneous coordinate ----------------------------------
def homogeneous_transfrom(points):
    # points - array of points
    # transformation given set of points to homogeneous - [x, y, w], w = 1
    return np.vstack((points, np.ones((1, points.shape[1]))))

# ---------------------------------- normalize ----------------------------------
def cnormalize_(pts):
    for p in pts:
        p /= pts[-1]
    return pts

# ---------------------------------- alpha triangle ----------------------------------
def alpha_for_triangle(pts, h, w):
    # alpha map for a triangle with corners in points(pts)
    alpha = np.zeros((h, w), np.uint8)
    
    for i in range(np.min(pts[0]), np.max(pts[0])):
        for j in range(np.min(pts[1]), np.max(pts[1])):
            x_ = np.linalg.solve(pts, [i, j, 1])
            if np.min(x_) > 0:
                alpha[i,j] = 1

    return alpha

# ---------------------------------- Delaunay triangulation ----------------------------------
def triangulate_points(x, y):
    """ Delaunay triangulation of 2D points. """
    size = x.shape[0]
    pts = np.array([[x[i], y[i]] for i in range(size)])
    triangles = Delaunay(pts)
    return triangles.simplices


# ---------------------------------- read points from xml ----------------------------------
def rd_pts_faces_xml(xmlfilepath, tag):
    
    xmlfile = minidom.parse(xmlfilepath)
    data_list = xmlfile.getElementsByTagName(tag)
    datas = {}
    for xmldata in data_list:
        fname = xmldata.attributes['file'].value 
        xf = np.int(xmldata.attributes['xf'].value)
        yf = np.int(xmldata.attributes['yf'].value)
        xs = np.int(xmldata.attributes['xs'].value)
        ys = np.int(xmldata.attributes['ys'].value)
        xm = np.int(xmldata.attributes['xm'].value)
        ym = np.int(xmldata.attributes['ym'].value)
        datas[fname] = np.array([xf, yf, xs, ys, xm, ym])
    return datas


# ---------------------------------- convert to skew-symmetric matrix ----------------------------------
def skew_sym(vec):
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])

# ---------------------------------- plot 2D boundary ----------------------------------
def plot2D_boundary(plot_range, pts, model, labels, values=[0]):
    # plot_range - interval(xmin, xmax, ymin, ymax), pts - list of points 
    # decisionfcn - deciding funciton, labels - array of class label, values - list of decision contours to show
    # list of colors for plot
    clr_list = ['b', 'r', 'g', 'k', 'm', 'y']

    # compute and plot on grid contour of decision function
    x = np.arange(plot_range[0], plot_range[1], 0.1)
    y = np.arange(plot_range[2], plot_range[3], 0.1)
    _x, _y = np.meshgrid(x, y)
    __x, __y = _x.flatten(), _y.flatten()
    _z = np.array(model.classify_2d(__x, __y)).reshape(_x.shape)

    pylab.contour(_x, _y, _z, values)

    # for every class draw a points: '*' - right, 'o' - wrong
    for i in range(len(pts)):
        d_ = model.classify_2d(pts[i][:, 0], pts[i][:, 1])
        correct_idx = labels[i] == d_
        incorrect_idx = labels[i] != d_
        pylab.plot(pts[i][correct_idx, 0], pts[i][correct_idx, 1], '*', color=clr_list[i])
        pylab.plot(pts[i][incorrect_idx, 0], pts[i][incorrect_idx, 1], 'o', color=clr_list[i])
    pylab.axis('equal')

    return

