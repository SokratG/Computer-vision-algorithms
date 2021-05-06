import numpy as np
import utilCV
import pylab
import cv2
import CVml
from smooth import ROF_denose
from mpl_toolkits.mplot3d import axes3d
import features
import transformation
import photography
import camera
import segmentation



def test_pca_():
    imgs_str = "imgs/data/a_thumbs/"
    namesamples = utilCV.get_dir_files(imgs_str, ".jpg")
    img_ = cv2.imread(namesamples[0], cv2.IMREAD_UNCHANGED)
   
    h, w = img_.shape
    l_imgs = np.array([cv2.imread(img_name, cv2.IMREAD_UNCHANGED) for img_name in namesamples])

    Mproj, var, mean = CVml.principal_component_analisys(l_imgs)
    import pickle

    with open('"imgs/data/a_thumbs/font_pca_modes.pkl', 'wb') as f:
        pickle.dump(mean, f)
        pickle.dump(Mproj, f)

    pylab.figure()
    pylab.gray()
    pylab.subplot(2, 4, 1)
    pylab.imshow(mean.reshape(h, w))
    # seven images with larger variance (mode statistics)
    for i in range(7):
        pylab.subplot(2, 4, i+2)
        pylab.imshow(Mproj[i].reshape(h, w))
    pylab.show()
    return 


def test_rof_():
    img_str = "imgs/test.jpg"
    img = cv2.imread(img_str, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    
    noise_img = utilCV.add_noise_r(img, sigma=10)


    U, T = ROF_denose(noise_img, noise_img, TV_weight=100)

    simg_U = "denoise_img"
    simg_T = "diff_img"
    utilCV.show_compare_r([img, noise_img, U, T], ['original', 'noise', simg_U, simg_T], 2, 2)

    return 



def test_harris_():
    img_str = "imgs/test.jpg"
    img = cv2.imread(img_str, cv2.IMREAD_GRAYSCALE)
   
    corns = features.corner_detector(img, sigma=3, min_dist=7, threshold=0.05)

    pylab.figure()
    pylab.gray()
    pylab.imshow(img)
    pylab.plot([p[1] for p in corns], [p[0] for p in corns], 'o', markersize=2)
    pylab.axis('off')
    pylab.show()

    return


def test_matches_Harris_():
    img_str1 = "imgs/data/crans_1_small.jpg"
    img_str2 = "imgs/data/crans_2_small.jpg"
    img1 = cv2.imread(img_str1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_str2, cv2.IMREAD_GRAYSCALE)
    
    px_width = 5

    Harris_coord_img1 = features.corner_detector(img1, sigma=5, min_dist=px_width+2, threshold=0.15)
    Harris_coord_img2 = features.corner_detector(img2, sigma=5, min_dist=px_width+2, threshold=0.15)

    desc_img1 = features.calc_desciptors(img1, Harris_coord_img1, px_width)
    desc_img2 = features.calc_desciptors(img2, Harris_coord_img2, px_width)

    # slow operation
    best_matches = features.find_best_matches(desc_img1, desc_img2) 
   
    
    pylab.figure()
    pylab.gray()
    
    res_img = utilCV.concat_imgs(img1, img2)
    pylab.imshow(res_img)
    
    offset_cols_ = img1.shape[1]
    for i, j in enumerate(best_matches):
        if (j > 0):
            pylab.plot([Harris_coord_img1[i][1], Harris_coord_img2[j][1] + offset_cols_],[Harris_coord_img1[i][0], Harris_coord_img2[j][0]], 'c')

    pylab.axis('off')
    
    pylab.show()
   
    return


def test_triangle_():
    x, y = np.array(np.random.standard_normal((2, 100)))
    
    tri = utilCV.triangulate_points(x, y)
    pylab.figure()
    for t in tri:
        t_ext = [t[0], t[1], t[2], t[0]]
        pylab.plot(x[t_ext], y[t_ext], 'r')
    pylab.plot(x, y, '*')
    pylab.axis('off')
    pylab.show()

    return



def test_warp_tri_():
    img_str1 = "imgs/data/sunset_tree.jpg"
    img_str2 = "imgs/data/turningtorso1.jpg"
    img1 = cv2.imread(img_str1, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(img_str2, cv2.IMREAD_UNCHANGED)
    
    x_, y_ = np.meshgrid(range(5), range(6))
    x = (img1.shape[1]/4) * x_.flatten()
    y = (img1.shape[0]/5) * y_.flatten()

    triangles = utilCV.triangulate_points(x, y)
    temp_pts = np.loadtxt('imgs/data/turningtorso1_points.txt', np.int32)
    
    src_pts = np.vstack((y, x, np.ones((1, len(x)))))
    proj_pts = np.int32(np.vstack((temp_pts[:,1], temp_pts[:,0], np.ones((1, len(temp_pts))))))
    
    #res = transformation.partial_warp_affine(img1, img2, src_pts, proj_pts, triangles)

    pylab.figure()
    pylab.imshow(img2)

    for tri_idx in triangles:
        t_ext = [tri_idx[0], tri_idx[1], tri_idx[2], tri_idx[0]] 
        pylab.plot(proj_pts[1][t_ext], proj_pts[0][t_ext],'g')

    pylab.axis('off')
    pylab.show()

    return


def test_imgreg_():
    filename = 'imgs/data/jkfaces.xml'
    pts = utilCV.rd_pts_faces_xml(filename, 'face') # read marker points face (eyes and mouth)
    _ = photography.rigid_alignment(pts, 'imgs/data/jkfaces/')

    return

def test_cameraobj_():
    pts = np.loadtxt('imgs/data/house.p3d').T
    pts = np.vstack((pts, np.ones(pts.shape[1])))
    
    # camera parameter
    P = np.hstack((np.eye(3), np.array([[0], [0], [-10]])))
    cam_ = camera.Camera(P)
    x = cam_.project(pts)
  
    # rotate camera around random axis
    r = 0.05 * np.random.rand(3)
    rot = cam_.rotation_matrix(r)
    test_it = 20
    pylab.figure()
    for i in range(test_it):
        cam_.P = np.dot(cam_.P, rot)
        x = cam_.project(pts)
        pylab.plot(x[0], x[1], 'k.')
    pylab.show()


    K = np.array([[1000, 0, 500], [0, 1000, 300], [0, 0, 1]])
    tmp = cam_.rotation_matrix([0, 0, 1])[:3, :3]
    Rt = np.hstack((tmp, np.array([[50], [40], [30]])))
    cam_2 = camera.Camera(np.dot(K, Rt))

    print(K)
    print(Rt)
    print(cam_2.get_cam_param())

    return



def prepare_mview_(num_view=3):
    
    # load 2d points from dataset
    points2D = [np.loadtxt('imgs/data/merton/00' + str(i+1) + '.corners').T for i in range(num_view)]

    # load 3d points from dataset
    points3D = np.loadtxt('imgs/data/merton/p3d').T

    # load correspondence
    corr_ = np.genfromtxt('imgs/data/merton/nview-corners', dtype=np.int32, missing_values='*')

    # load camera matrix
    P = [camera.Camera(np.loadtxt('imgs/data/merton/00' + str(i+1) + '.P')) for i in range(num_view)]

    return points2D, points3D, corr_, P



def test_3d_():
    fig = pylab.figure()
    ax = fig.gca(projection='3d')
    X,Y,Z = axes3d.get_test_data(0.25)
    ax.plot(X.flatten(), Y.flatten(), Z.flatten(), 'o')
    pylab.show()
    return

def test_epipolar_pts_():
    NUM_VIEW = 3
    img1 = cv2.imread('imgs/data/merton/001.jpg', cv2.IMREAD_UNCHANGED)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread('imgs/data/merton/002.jpg', cv2.IMREAD_UNCHANGED)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    points2D, points3D, corr, P = prepare_mview_()

    pts = np.vstack((points3D, np.ones(points3D.shape[1])))
    p = P[0].project(pts)

    pylab.figure()
    pylab.imshow(img1)
    pylab.plot(points2D[0][0], points2D[0][1], 'o', markersize=3)
    pylab.axis('off')

    pylab.figure()
    pylab.imshow(img1)
    pylab.plot(p[0], p[1], 'r.')
    pylab.axis('off')


    fig = pylab.figure()
    ax = fig.gca(projection='3d')
    ax.plot(points3D[0], points3D[1], points3D[2], 'k.')

    pylab.show()
    return


def test_epipole_():
    EPIPOLE_NUM = 5
    img1 = cv2.imread('imgs/data/merton/001.jpg', cv2.IMREAD_UNCHANGED)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread('imgs/data/merton/002.jpg', cv2.IMREAD_UNCHANGED)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    points2D, points3D, corr, P = prepare_mview_()
    idx = (corr[:, 0] >= 0) & (corr[:, 1] > 0)
    pts1 = utilCV.homogeneous_transfrom(points2D[0][:, corr[idx, 0]])
    pts2 = utilCV.homogeneous_transfrom(points2D[1][:, corr[idx, 1]])

    # fundamental matrix
    F = transformation.find_fundamental_matrix(pts1, pts2)
    # epipole
    e = transformation.find_epipole(F)

    # draw epipole
    pylab.figure()
    pylab.imshow(img1)
    h, w, _ = img1.shape
    
    for i in range(EPIPOLE_NUM):
        line = np.dot(F, pts2[:, i])
        param_line = np.linspace(0, w, 100)
        line_val = np.array([(line[2] + line[0]*pl)/(-line[1]) for pl in param_line])

        # points of straight line inside a image
        idx = (line_val >= 0) & (line_val < h)
        pylab.plot(param_line[idx], line_val[idx], linewidth=2)
        # show epipole
        #pylab.plot(e[0]/e[2], e[1]/e[2], 'r*')

    pylab.axis('off')
    
    pylab.figure()
    pylab.imshow(img2)
    for i in range(EPIPOLE_NUM):
        pylab.plot(pts2[0, i], pts2[1, i], 'o')
    pylab.axis('off')
    
    pylab.show()

    return




def test_stereo_depth_():
    img1 = cv2.imread('imgs/data/tsukuba/scene1.row3.col3.ppm', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('imgs/data/tsukuba/scene1.row3.col4.ppm', cv2.IMREAD_GRAYSCALE)
    
    steps = 12
    start = 3
    # width of block norm cross-correlation
    width = 9
    res_img = photography.plane_sweep_ncc(img1, img2, start, steps, width)
    res_img_g = photography.plane_sweep_ncc_gauss(img1, img2, start, steps, width=3)

    fig, axes = pylab.subplots(nrows=1, ncols=2) 
    pylab.gray()
    pylab.axis('off')
    axes[0].imshow(res_img)
    axes[1].imshow(res_img_g)
    pylab.show()
 
    return


def test_cluster_():
    import scipy.cluster.vq
    class1 = 1.5 * np.random.randn(100, 2)
    class2 = np.random.randn(100, 2) + np.array([5, 5])
    feature = np.vstack((class1, class2))

    centr, var = scipy.cluster.vq.kmeans(feature, 2)
    code, dist = scipy.cluster.vq.vq(feature, centr)

    pylab.figure()
    idx = np.where(code == 0)[0]
    pylab.plot(feature[idx, 0], feature[idx, 1], '*')
    idx = np.where(code == 1)[0]
    pylab.plot(feature[idx, 0], feature[idx, 1], 'r.')
    pylab.plot(centr[:, 0], centr[:, 1], 'go')
    pylab.show()

    return


def test_cluster_font_():
    K = 4
    imgs_str = "imgs/data/a_selected_thumbs/"
    namesamples = utilCV.get_dir_files(imgs_str, ".jpg") 
    img_len = len(namesamples)
    '''
    import pickle
    with open('imgs/data/a_selected_thumbs/a_pca_modes.pkl', 'rb') as f:
        imgmean_ = pickle.load(f)
        V = pickle.load(f)
    '''

     
    ll_imgs = np.array([cv2.imread(img_name, cv2.IMREAD_UNCHANGED) for img_name in namesamples])
    V, var, imgmean_ = CVml.principal_component_analisys(ll_imgs)
    imgmean = imgmean_.flatten()
    
    # linear images
    ll_imgs = np.array([img.flatten() for img in ll_imgs], np.float32)


    # project to first 40 component
    projected_img = np.array([np.dot(V[:40], ll_imgs[i] - imgmean) for i in range(img_len)])
    import scipy.cluster.vq
    projected = scipy.cluster.vq.whiten(projected_img) # normalize feature variance to 1
    
    centroid, distortion = scipy.cluster.vq.kmeans(projected, K)
    
    code, distance = scipy.cluster.vq.vq(projected, centroid)

    # draw result
    for k in range(K):
        idx = np.where(code == k)[0] 
        pylab.figure()
        pylab.gray()
        for i in range(np.minimum(len(idx), 40)):
            pylab.subplot(K, 10, i+1)
            pylab.imshow(ll_imgs[idx[i]].reshape((25, 25)))
            pylab.axis('off')
    pylab.show()

    return


def test_cluster_pixel_():
    steps = 50 # step block
    K = 3 # r g b
    img_ = cv2.imread('imgs/data/empire.jpg', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    # divide image on area
    dx_ = int(img.shape[0] / steps)
    dy_ = int(img.shape[1] / steps)
    import scipy.cluster.vq
    import skimage.transform
    
    # calc for color features for area
    features = []
    for x in range(steps):
        for y in range(steps):
            R = np.mean(img[x*dx_:(x+1)*dx_, y*dy_:(y+1)*dy_, 0])
            G = np.mean(img[x*dx_:(x+1)*dx_, y*dy_:(y+1)*dy_, 1])
            B = np.mean(img[x*dx_:(x+1)*dx_, y*dy_:(y+1)*dy_, 2])
            features.append([R,G,B])
    
    features = np.array(features, np.float32)

    centroid, distortion = scipy.cluster.vq.kmeans(features, K)
    code, distance = scipy.cluster.vq.vq(features, centroid)

    codeimg_ = code.reshape(steps, steps)
    #codeimg = cv2.resize(codeimg_, img.shape[:2], interpolation=cv2.INTER_NEAREST) #scipy.misc.imresize(codeimg_, img.shape[:2], interp='nearest')
    codeimg = skimage.transform.resize(codeimg_, img.shape[:2], order=0) 
    pylab.figure()
    pylab.imshow(codeimg)
    pylab.show()

    return


def test_hierarchical_cluster_():
    import scipy.cluster.vq
    class1 = 1.5 * np.random.randn(100, 2)
    class2 = np.random.randn(100, 2) + np.array([5, 5])
    features = np.vstack((class1, class2))

    tree = CVml.hierarchical_cluster(features)

    clusters = tree.extract_cluster(5)
    print('Number of clusters: ' + str(len(clusters)))
    for cl in clusters:
        print(cl.get_cluster_elements())

    return


def test_spectral_cluster_():
    imgs_str = "imgs/data/a_selected_thumbs/"
    namesamples = utilCV.get_dir_files(imgs_str, ".jpg") 
    img_len = len(namesamples)
    l_imgs = np.array([cv2.imread(img_name, cv2.IMREAD_UNCHANGED) for img_name in namesamples])

    # first k's eigenvectors
    k = 5
    _, _, code, _ = CVml.spectral_cluster(l_imgs, img_len, k)
    
    
    for cl in range(k):
        idx = np.where(code == cl)[0]
        pylab.figure()
        for i in range(np.minimum(len(idx), 39)):
            img_ = cv2.imread(namesamples[idx[i]], cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
            pylab.subplot(4, 10, i+1)
            pylab.imshow(img)
            pylab.axis('equal')
            pylab.axis('off')
    pylab.show()
    return



def test_knn_generate_data_():
    N = 200

    # normal distribution
    class_1 = 0.6 * np.random.randn(N, 2)
    class_2 = 1.2 * np.random.randn(N, 2) + np.array([5, 1])
    labels = np.hstack((np.ones(N), -np.ones(N)))
    import pickle
    with open('imgs/data/points_normal.pkl', 'wb') as f:
        pickle.dump(class_1, f)
        pickle.dump(class_2, f)
        pickle.dump(labels, f)
    
    # normal distribution and ring around them
    class_1 = 0.6 * np.random.randn(N, 2)
    r = 0.8 * np.random.randn(N, 1) + 5
    angle = 2*np.pi + np.random.randn(N, 1)
    class_2 = np.hstack((r * np.cos(angle), r * np.sin(angle)))
    labels = np.hstack((np.ones(N), -np.ones(N)))
    with open('imgs/data/points_ring.pkl', 'wb') as f:
        pickle.dump(class_1, f)
        pickle.dump(class_2, f)
        pickle.dump(labels, f)
    return

def test_knn_():
    import pickle
    with open('imgs/data/points_normal.pkl', 'rb') as f:
        class_1 = pickle.load(f)
        class_2 = pickle.load(f)
        labels = pickle.load(f)
    knnmodel = CVml.KnnClassifier(labels, np.vstack((class_1, class_2)))
    with open('imgs/data/points_normal_test.pkl', 'rb') as f:
        class_1 = pickle.load(f)
        class_2 = pickle.load(f)
        labels = pickle.load(f)

    utilCV.plot2D_boundary([-6, 6, -6, 6], [class_1, class_2], knnmodel, [1, -1])
    pylab.show()
    return


def prepare_gesture_data(gesture_dir, type_dir, pathname="imgs/data/hog_data/", template_size=(50, 50)):

    size_d = len(gesture_dir)
    features_ = []
    labels = []
    # prepare train data set
    for i in range(size_d):
        sub_dirs = utilCV.get_dir_files(pathname+gesture_dir[i]+type_dir, ".ppm") 
        for str_img in sub_dirs:
            img = cv2.resize(cv2.cvtColor(cv2.imread(str_img, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), template_size)
            desc = features.hog_descs(img)
            features_.append(desc.flatten())
            labels.append(str_img.split('/')[-1][0])
    
    features_ = np.array(features_)
    labels = np.array(labels)

    return features_, labels

def test_hog_knn_classify_():
    
    dir_ = "imgs/data/hog_data/"
    gesture_dir = ['A/', 'B/', 'C/', 'Five/', 'Point/', 'V/']
    template_size = (50, 50)
    size_d = len(gesture_dir)

    # prepare train data set
    type_dir = 'train/'
    features_, labels =  prepare_gesture_data(gesture_dir, type_dir, dir_, template_size)     
    
    # prepare test data set
    type_dir = 'test/'
    test_features_, test_labels = prepare_gesture_data(gesture_dir, type_dir, dir_, template_size)
    
   
    # classify
    classnames = np.unique(labels)
    nbr_classes = len(classnames)
    knn_classifier = CVml.KnnClassifier(labels, features_)
    K = 1
    res = np.array([knn_classifier.classify(test_features_[i], K) for i in range(len(test_labels))])

    # accuracity
    acc = np.sum(1.0 * (res == test_labels)) / len(test_labels)
    print("Accuracity = %f" % acc)
    class_ind = dict([(classnames[i], i) for i in range(nbr_classes)])
    confuse = np.zeros((nbr_classes, nbr_classes))
    for i in range(len(test_labels)):
        confuse[class_ind[res[i]], class_ind[test_labels[i]]] += 1
    print('Confuse matrix:')
    print(classnames)
    print(confuse)

    return 

def test_bayes_():
    import pickle
    with open('imgs/data/points_normal.pkl', 'rb') as f:
        class_1 = pickle.load(f)
        class_2 = pickle.load(f)
        labels = pickle.load(f)
    bsmodel = CVml.BayesClassifier()
    bsmodel.train([class_1, class_2], [1, -1])
    with open('imgs/data/points_normal_test.pkl', 'rb') as f:
        class_1 = pickle.load(f)
        class_2 = pickle.load(f)
        labels = pickle.load(f)
    print(bsmodel.classify(class_1[:10])[0])
    utilCV.plot2D_boundary([-6, 6, -6, 6], [class_1, class_2], bsmodel, [1, -1])
    pylab.show()
    return

def test_bayes_classifier_():
    dir_ = "imgs/data/hog_data/"
    gesture_dir = ['A/', 'B/', 'C/', 'Five/', 'Point/', 'V/']
    template_size = (50, 50)
    size_d = len(gesture_dir)

    # prepare train data set
    type_dir = 'train/'
    features_, labels =  prepare_gesture_data(gesture_dir, type_dir, dir_, template_size) 
    type_dir = 'test/'
    test_features_, test_labels = prepare_gesture_data(gesture_dir, type_dir, dir_, template_size)
    
    V_, _, mean = CVml.principal_component_analisys(features_)

    # DON'T WORK! check covariance matrix (eigenvalue and vec contain 'nan')!!!!
    V = V_[:50] 
     
    features_ = np.array([np.dot(V, feature - mean) for feature in features_])
    test_features = np.array([np.dot(V, feature - mean) for feature in test_features_])
    
    # classify
    classnames = np.unique(labels)
    nbr_classes = len(classnames)

    bsmodel = CVml.BayesClassifier()
    bslist = [features_[np.where(labels==cl)[0]] for cl in classnames]
    bsmodel.train(bslist, classnames)
    res = bsmodel.classify(test_features)[0]
    
    # calc accuracity
    acc = np.sum(1.0 * (res == test_labels)) / len(test_labels) # ?
    print("Accuracity = %f" % acc)
    class_ind = dict([(classnames[i], i) for i in range(nbr_classes)])
    confuse = np.zeros((nbr_classes, nbr_classes))
    for i in range(len(test_labels)):
        confuse[class_ind[res[i]], class_ind[test_labels[i]]] += 1
    print('Confuse matrix:')
    print(classnames)
    print(confuse)
    return


def test_svm_():
    import pickle
    import libsvm.svmutil as svmutil
    with open('imgs/data/points_normal.pkl', 'rb') as f:
        class_1 = np.array(pickle.load(f))
        class_2 = np.array(pickle.load(f))
        labels = pickle.load(f)
    
    samples = class_1.tolist() + class_2.tolist()
    
    # make svm classifier
    prob = svmutil.svm_problem(labels, samples)
    param = svmutil.svm_parameter('-t 2')

    # train svm-classifier
    m_ = svmutil.svm_train(prob, param)

    res = svmutil.svm_predict(labels, samples, m_)

    with open('imgs/data/points_normal_test.pkl', 'rb') as f:
        class_1 = np.array(pickle.load(f))
        class_2 = np.array(pickle.load(f))
        labels = pickle.load(f)

    class Predict(object):
        def __init__(self):
            pass
        def classify_2d(self, x, y, model=m_):
            pack = list(zip(x, y))
            return np.array(svmutil.svm_predict([0]*len(x), pack, model)[0])

    utilCV.plot2D_boundary([-6, 6, -6, 6], [class_1, class_2], Predict(), [-1, 1])
    pylab.show()
    return


def test_svm_classifier_():
    dir_ = "imgs/data/hog_data/"
    gesture_dir = ['A/', 'B/', 'C/', 'Five/', 'Point/', 'V/']
    template_size = (50, 50)
    size_d = len(gesture_dir)

    # prepare train data set
    type_dir = 'train/'
    features_, labels =  prepare_gesture_data(gesture_dir, type_dir, dir_, template_size) 
    features_ = features_.tolist()
    type_dir = 'test/'
    test_features_, test_labels = prepare_gesture_data(gesture_dir, type_dir, dir_, template_size)
    test_features_ = test_features_.tolist()

    import libsvm.svmutil as svmutil
    # classify
    classnames = np.unique(labels)
    nbr_classes = len(classnames)

    # function for label transformation
    transl = {}
    for i, cl in enumerate(classnames):
        transl[cl], transl[i] = i, cl
    

    # make svm classifier
    def convert_labels(labels, transl):
        # Convert between strings and numbers. 
        return [transl[l] for l in labels]

    prob = svmutil.svm_problem(convert_labels(labels, transl), features_)
    param = svmutil.svm_parameter('-t 0')

    # train svm-classifier
    svmmodel = svmutil.svm_train(prob, param)

    res = svmutil.svm_predict(convert_labels(labels, transl), features_, svmmodel)

    # check test set
    res = svmutil.svm_predict(convert_labels(test_labels, transl), test_features_, svmmodel)[0]
    res = convert_labels(res, transl)

    # calc accuracity
    acc = np.sum(1.0 * (res == test_labels)) / len(test_labels) 
    print("Accuracity = %f" % acc)
    class_ind = dict([(classnames[i], i) for i in range(nbr_classes)])
    confuse = np.zeros((nbr_classes, nbr_classes))
    for i in range(len(test_labels)):
        confuse[class_ind[res[i]], class_ind[test_labels[i]]] += 1
    print('Confuse matrix:')
    print(classnames)
    print(confuse)

    return


def test_graph_cut_segmentation_():
    # segmentation
    import scipy.misc 
    img_ = cv2.imread('imgs/data/empire.jpg', cv2.IMREAD_UNCHANGED)

    img = cv2.resize(cv2.cvtColor(img_, cv2.COLOR_BGR2RGB), None, fx=0.07, fy=0.07, interpolation=cv2.INTER_LINEAR)
    size_ = img.shape[:2]
    
    # make a train area labels for bayes
    labels = np.zeros(size_)
    labels[3:18, 3:18] = -1
    labels[-18:-3, -18:-3] = 1

    # build a bayes graph
    graph_ = segmentation.build_bayes_graph(img, labels, kappa=1)

    # cut a graph
    resimg = segmentation.graph_cut(graph_, size_)

    pylab.figure()
    utilCV.draw_labels(img, labels)
    pylab.figure()
    pylab.imshow(resimg)
    pylab.gray()
    pylab.axis('off')
    pylab.show()
    
    return


def test_cluster_segmentation_():
    img_ = cv2.imread('imgs/data/C-uniform03.ppm', cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    width = 50
    img = np.float32(cv2.resize(img, (width, width), interpolation=cv2.INTER_LINEAR))
    
    # make normalizing cut matrix 
    A = segmentation.norm_cut_graph(img, sigma_space=1, sigma_color=0.01)

    # cluster
    code, _ = segmentation.spectral_cluster_cut_segmentation(A, k=3, ndim=3)

    rimg = cv2.resize(code.reshape(width, width), (h, w), interpolation=cv2.INTER_NEAREST)

    pylab.figure()
    pylab.imshow(rimg)
    pylab.gray()
    pylab.axis('off')
    pylab.show()

    return

def test_cartoon_():
    img_str = "imgs/test2.jpg"
    img = cv2.imread(img_str, cv2.IMREAD_UNCHANGED)
    rimg = photography.cartoon_animate(img)
    cv2.imshow('win', rimg)
    cv2.waitKey(0)
    return
    
def test_dft_():
    import dft
    img_str = "imgs/messi.jpeg"
    img = cv2.imread(img_str, cv2.IMREAD_GRAYSCALE)
    res = dft.DFT_img(img, True)
    
    cv2.imwrite('res.png', res)
    return
    
def test_laplacian_():
    import smooth
    img_str = "imgs/moon.png"
    img = cv2.imread(img_str, cv2.IMREAD_GRAYSCALE)
    res = smooth.customLaplaceSharpening(img)
    cv2.imshow('win', res)
    cv2.imshow('win2', img)
    cv2.waitKey(0)
    return
  
def test_LoG_():
    import edge_detection
    img_str = "imgs/Set1Seq1.bmp"
    img = cv2.imread(img_str, cv2.IMREAD_GRAYSCALE)
    res = edge_detection.LoG_edge(img)
    cv2.imshow('win', res)
    cv2.waitKey(0)
    return  
    
