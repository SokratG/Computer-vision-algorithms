import numpy as np
from itertools import combinations
import cv2

# ---------------------------------- Principal Component Analisys ----------------------------------
def principal_component_analisys(l_imgs):
    # l_imgs - list of images
    
    # 2D matrix(linear images), one row - one image (numpy flatten)
    linear_imgs = np.array([img.flatten() for img in l_imgs], np.float32)
    
    data_size, dim = linear_imgs.shape
 
    mean_data =  linear_imgs.mean(axis=0)
    linear_imgs = linear_imgs - mean_data
    
    # calc PCA 
    if (dim > data_size): # for large dimensions 
        covarianceMatrix = np.cov(linear_imgs, bias=True)    
        evalue, evector = np.linalg.eigh(covarianceMatrix)
        mat_svd = np.dot(linear_imgs.T, evector).T
        # change order
        Mproj = mat_svd[::-1]
        var = np.sqrt(evalue)[::-1]
        for i in range(Mproj.shape[1]):
            Mproj[:, i] /= var
    else:
        # SVD decomposition
        Up, var, Mproj = np.linalg.svd(linear_imgs)
        Mproj = Mproj[:data_size] # only firsts data_size datas

    # return Matrix projection for multidimensional matrix, variance of data, mean of data
    return Mproj, var, mean_data

# ---------------------------------- Hierarchical Clustering ----------------------------------
class ClusterNode(object):
    def __init__(self, vec, l_node, r_node, distance=0.0, count=1):
        self.vec = vec
        self.right = r_node
        self.left = l_node
        self.distance = distance
        self.count = count
    def extract_cluster(self, distance):
        # extract sub-tree from tree hier. clusters, when self distance less than given distance 
        if self.distance < distance:
            return [self]
        return self.left.extract_cluster(distance) + self.right.extract_cluster(distance)
    def get_cluster_elements(self):
        # return id of nodesin sub-tree of h. cluster
        return self.left.get_cluster_elements() + self.right.get_cluster_elements()
    def get_height(self):
        # return height of node
        return self.left.get_height() + self.right.get_height()
    def get_depth(self):
        # return a depth nodes, max of depth sub nodes + distance
        return np.max(self.left.get_depth() + self.right.get_depth()) + distance

class ClusterLeafNode(object):
    def __init__(self, vec, id):
        self.vec = vec
        self.id = id
    def extract_cluster(self, dist):
        return [self]
    def get_cluster_elements(self):
        return [self.id]
    def get_height(self):
        return 1
    def get_depth(self):
        return 0

def L2norm_(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))

def L1norm_(vec1, vec2):
    return np.sum(np.abs(vec1-vec2))

def hierarchical_cluster(features, distfunc=L2norm_):
    # to cluster features using method by  hierarchical clusters

    distances = {}
    nodes_ = [ClusterLeafNode(np.array(feature), id=idx) for idx, feature in enumerate(features)]

    while (len(nodes_) > 1):
        closest = np.Inf
        # iterate all node pairs and find minimal distance
        for i, j in combinations(nodes_, 2):
            if (i, j) not in distances:
                distances[i, j] = distfunc(i.vec, j.vec)
            d = distances[i, j]
            if d < closest:
                closest = d
                lowestpair = (i, j)
        node_i, node_j = lowestpair
        # average two clusters
        n_vec = (node_i.vec + node_j.vec) / 2.
        
        # make a new node
        n_node = ClusterNode(n_vec, l_node=node_i, r_node=node_j, distance=closest)
        nodes_.remove(node_i)
        nodes_.remove(node_j)
        nodes_.append(n_node)

    # return root node
    return nodes_[0]


# ---------------------------------- Spectral Clustering ----------------------------------
def spectral_cluster(l_imgs, img_len, K=2, pca_num=40):

    V, var, imgmean_ = principal_component_analisys(l_imgs)
    imgmean = imgmean_.flatten()
    
    # linear images
    ll_imgs = np.array([img.flatten() for img in l_imgs], np.float32)
    # project to first 40 component
    projected_img = np.array([np.dot(V[:pca_num], ll_imgs[i] - imgmean) for i in range(img_len)])
    import scipy.cluster.vq
    projected = scipy.cluster.vq.whiten(projected_img) # need normalize?

    size_ = len(projected)
    # sum of matrix distance
    S = np.array([[np.sqrt(np.sum((projected[i] - projected[j])**2)) for i in range(size_)] for j in range(size_)], np.float32)

    # make a Laplace matrix
    rowsum = np.sum(S, axis=0)
    D = np.diag(1 / np.sqrt(rowsum)) # D^(-1/2)
    I = np.identity(size_)
    L = I - np.dot(D, np.dot(S, D))

    # calc eigenvector
    U, S, V = np.linalg.svd(L)
    
    features_ = np.array(V[:K]).transpose()
    features_ = scipy.cluster.vq.whiten(features_)
    centroid, distortion = scipy.cluster.vq.kmeans(features_, K)
    code, distance = scipy.cluster.vq.vq(features_, centroid)

    return features_, centroid, code, distance


# ---------------------------------- KNN classification ----------------------------------
class KnnClassifier(object):
    def __init__(self, labels, samples, classifymetric=L2norm_):
        self.labels = labels
        self.samples = samples
        self.classifymetric = classifymetric

    def classify(self, pt, k=3):
        # classify point on k neighbor points in train set

        # calc distance from all points in train set
        dist_ = np.array([self.classifymetric(pt, sample) for sample in self.samples])
        idx = dist_.argsort()
        # save k nearest in dictionary
        votes = {}
        for i in range(k):
            label = self.labels[idx[i]]
            votes.setdefault(label, 0)
            votes[label] += 1
        # return label
        return max(votes)

    def classify_2d(self, x, y):
        return np.array([self.classify([x_, y_]) for (x_, y_) in zip(x, y)])

# ---------------------------------- Bayes classification ----------------------------------
def gauss_(mean, var, x):
    # compute d-dimentional normal distribution with mean and varicance in points 'x'
    if len(x.shape)==1:
        num, dim = 1, x.shape[0]
    else:
        num, dim = x.shape
    # covariance matrix
    S = np.diag(1/var)
    x_ = x - mean
    y = np.exp(-0.5 * np.diag(np.dot(x_, np.dot(S, x_.T))))

    # return normalize result
    return y * (2 * np.pi)**(-dim/2.0) / (np.sqrt(np.prod(var)) + 1e-6)

class BayesClassifier(object):
    def __init__(self):
        # init labels of class, mean of class, variance of class, number of class 
        self.labels = []
        self.mean = []
        self.var = []
        self.num = 0

    def train(self, data, labels=None):
        # train data 
        if labels is None:
            labels = range(len(data))
        self.labels = labels
        self.num = len(labels)
        for d in data:
            self.mean.append(np.mean(d, axis=0))
            self.var.append(np.var(d, axis=0))

    def classify(self, pts):
        # classify points - calc probability every class and return label with higher probability class
        # calc probality every class
        estimate_pr = np.array([gauss_(m, v, pts) for m, v in zip(self.mean, self.var)])
        # get index with high prob.
        idx = estimate_pr.argmax(axis=0)
        estimate_labels = np.array([self.labels[i] for i in idx])

        return estimate_labels, estimate_pr

    def classify_2d(self, x, y):
        pts = np.vstack((x, y))
        return self.classify(pts.T)[0]