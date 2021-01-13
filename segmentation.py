import numpy as np
import cv2
import networkx
import CVml


# ---------------------------------- Graph cut segmentation ----------------------------------
def build_bayes_graph(img, labels, sigma=10, kappa=2):
    # build a graph on 4-connection components(pixels).
    # front and back define on label - 1 - front, -1 - back, 0 - otherwise
    h, w = img.shape[:2]
    # rgb vector (by 1 px on row) 
    vimg = img.reshape((-1, 3))

    # rgb for fron and back
    front_ = img[labels == 1].reshape((-1, 3))
    back_ = img[labels == -1].reshape((-1, 3))
    train_data = [front_, back_]


    # make bayes classifier
    bsmodel = CVml.BayesClassifier()
    bsmodel.train(train_data)

    # get probality for every pixels
    bs_labes, px_prob = bsmodel.classify(vimg)
    prob_front_ = px_prob[0]
    prob_back_ = px_prob[1]
    
    # prepare a graph (h*w+2) 
    graph_ = networkx.DiGraph()
    graph_.add_nodes_from(range(h * w + 2))

    src_ = h * w # source index - pre-last node
    sink_ = h * w + 1 # last node - sink index

    # normalize
    for i in range(vimg.shape[0]):
        vimg[i] = vimg[i] / np.linalg.norm(vimg[i])

    # build a graph
    for i in range(h*w):
        # add edge from source
        wt_ = (prob_front_[i]/(prob_front_[i]+prob_back_[i]))
        graph_.add_edge(src_, i,  capacity=wt_, weight=wt_)

        # add edge to sink
        wt_ = (prob_back_[i]/(prob_front_[i]+prob_back_[i]))
        graph_.add_edge(i, sink_, capacity=wt_, weight=wt_)

        # add edges with neighbors (4 - connection components)
        if (i % w) != 0: # left neighbors
            wt_ = kappa * np.exp(-1.0 * np.sum((vimg[i] - vimg[i-1])**2) / sigma)
            graph_.add_edge(i, i-1, capacity=wt_, weight=wt_)
        if ((i+1) % w) != 0: # right neighbors
            wt_ = kappa * np.exp(-1.0 * np.sum((vimg[i] - vimg[i+1])**2) / sigma)
            graph_.add_edge(i, i+1, capacity=wt_, weight=wt_)
        if (i // w) != 0: # top neighbors
            wt_ = kappa * np.exp(-1.0 * np.sum((vimg[i] - vimg[i-w])**2) / sigma)
            graph_.add_edge(i, i-w, capacity=wt_, weight=wt_)
        if (i // w) != (h-1): # bottom neighbors
            wt_ = kappa * np.exp(-1.0 * np.sum((vimg[i] - vimg[i+w])**2) / sigma)
            graph_.add_edge(i, i+w, capacity=wt_, weight=wt_)
    # return building graph
    return graph_

def graph_cut(graph, imgsize):
    # find maximum graph flow and return binary img, composing from labels segmentation pixels
    h, w = imgsize
    src_ = h * w
    sink_ = h * w + 1
    
    gcut_value, gcut_p = networkx.algorithms.flow.minimum_cut(graph, src_, sink_)
    reachable, non_reachable = gcut_p
    cutset = set()
    for u, nbrs in ((n, graph[n]) for n in reachable):
        if u == h*w: # how avoid this ?
            continue
        cutset.update((u, v) for v in nbrs if v in non_reachable)
    
    res_ = np.zeros(h * w)
    for i, j in cutset:
        res_[i] = j
    
    return res_.reshape((h, w))


# ---------------------------------- Normalization cut(Clusterization) segmentation ----------------------------------
def norm_cut_graph(img, sigma_space=10, sigma_color=0.1):
    # img - image, sigma space, sigma color
    # return normalize cutting matrix with weights(distance pixels and pixel similarity)
    h, w = img.shape[:2]
    isColor = len(img.shape) == 3
    # normalize and make a vector features: RGB or grayscale
    img_ = img.copy()
    if isColor:
        for i in range(len(img.shape)):
            img_[:, :, i] /= img_[:, :, i].max()
        vimg = img_.reshape((-1, 3))
    else:
        img_ /= img_.max()
        vimg = img_.flatten()
    
    # coordinate for computer distance
    x_, y_ = np.meshgrid(range(h), range(w))
    x, y = x_.flatten(), y_.flatten()

    # create a matrix with edge weight
    N = h * w
    W = np.zeros((N, N), np.float32)
    for i in range(N):
        for j in range(N):
            d = (x[i] - x[j])**2 + (y[i]-y[j])**2
            W[i, j] = W[j, i] = np.exp(-1.0 * np.sum((vimg[i] - vimg[j])**2) / sigma_color) * np.exp(-d / sigma_space)

    return W

def spectral_cluster_cut_segmentation(S, k, ndim):
    # S - matrix of similarity, ndim - number of eigenvectors, k -number of clusters
    # spectral clusterization
    import scipy.cluster.vq

    # check symmetric matrix
    if np.sum(np.abs(S-S.T)) > 1e-9:
        print("non symmetric")
    # make Laplace matrix
    rowsum = np.sum(np.abs(S), axis=0)
    D = np.diag(1 / np.sqrt(rowsum + 1e-6))
    L = np.dot(D, np.dot(S, D))

    # find eigenvectors
    _, _, V = np.linalg.svd(L)

    # create vector of features from first ndim eigenvectors
    # cluster K-mean
    features = scipy.cluster.vq.whiten(np.array(V[:ndim]).transpose())
    centroids, _ = scipy.cluster.vq.kmeans(features, k)
    code, _ = scipy.cluster.vq.vq(features, centroids)

    return code, V