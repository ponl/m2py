import os
from collections import Counter

import numpy as np
import seaborn as sb
from scipy import signal
from matplotlib import pyplot
from sklearn.mixture import GaussianMixture

from methods import config

PROPS = config.data_properties
HEIGHT_INDEX = config.height_index

## Boundary Detection
def sobel(x):
    """ Applies the Sobel operator to x
    Args:
        x (np array): data
    Returns:
        (np array): data after applying convolution
    """
    fr = np.array([[0, 1, 0, -1, 0], [1, 2, 0, -2, -1], [2, 4, 0, -4, 2], [ 1,  2,  0, -2,  1], [0,  1,  0, -1, 0]])
    fc = np.array([[0, 1, 2,  1, 0], [1, 2, 4,  2,  1], [0, 0, 0,  0, 0], [-1, -2, -4, -2, -1], [0, -1, -2, -1, 0]])
    yr = signal.convolve2d(x, fr, boundary="symm", mode="same")
    yc = signal.convolve2d(x, fc, boundary="symm", mode="same")

    return np.sqrt(yr**2 + yc**2)

def show_boundaries(X):
    """ Plots the data properties and their boundaries
    Args:
        X (np array): data
    """
    fig = pyplot.figure(figsize=(16, 30), dpi= 80, facecolor='w', edgecolor='k')
    cnt = 1
    for i in range(X.shape[2]):
        x = X[:,:,i]
        pyplot.subplot(3,4,cnt) # TODO how to ensure break down of plots?
        pyplot.title(PROPS[i])
        m = pyplot.imshow(x)
        pyplot.colorbar(m, fraction=0.046, pad=0.04)

        cnt += 1

        # Apply Sobel operator
        pyplot.subplot(3,4,cnt)
        pyplot.title(f"{PROPS[i]} Gradient")
        m = pyplot.imshow(sobel(x))
        pyplot.colorbar(m, fraction=0.046, pad=0.04)

        cnt += 1

    pyplot.tight_layout()
    pyplot.show()

## Properties Distributions
def show_property_distributions(X, O):
    """ Plots the pdfs of the data properties
    Args:
        X (np array): data
        O (np array): outliers
    """
    fig = pyplot.figure(figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')

    for j in range(X.shape[2]):
        x = [X[n,m,j] for n in range(X.shape[0]) for m in range(X.shape[1]) if not O[n,m]]
        pyplot.subplot(2, 3, j+1) # TODO how to ensure break down of plots?
        pyplot.title(PROPS[j])
        sb.distplot(x) # TODO warning gets thrown out when using this
        pyplot.grid()

    pyplot.show()

## Outlier Detection
def extract_outliers(X, height_index=HEIGHT_INDEX, threshold=2.5):
    """ Finds outliers from data
    Args:
        X (np array): data
        height_index (int): index of height property
        threshold: (float): z-score threshold
    Returns:
        (np array): boolean matrix denoting outliers
    """
    x = X[:,:,height_index]

    # Smooth data
    flt = np.array([[0.5, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 0.5]])
    y = signal.convolve2d(x, flt, boundary='symm', mode='same')

    # Compute z-scores
    u = np.mean(y)
    s = np.std(y)
    z = np.abs((u-y)/s)

    # Threshold by z-score
    return z > threshold

def show_outliers(X, O, height_index=HEIGHT_INDEX):
    """ Plots data properties and outliers
    Args:
        X (np array): data
        O (np array): outliers
        height_index (int): index of height property
    """
    fig = pyplot.figure(figsize=(18,5))

    pyplot.subplot(1,3,1)
    m = pyplot.imshow(X[:,:,height_index], aspect="auto")
    pyplot.title('Height')
    pyplot.colorbar(m, fraction=0.046, pad=0.04)

    pyplot.subplot(1,3,2)
    pyplot.imshow(O, aspect='auto')
    pyplot.title('Height Outliers')

    X[O == 1] = 0
    pyplot.subplot(1,3,3)
    m = pyplot.imshow(X[:,:,height_index], aspect='auto')
    pyplot.title('Height')
    pyplot.colorbar(m, fraction=0.046, pad=0.04)

    pyplot.tight_layout()
    pyplot.show()

## Features Correlations
def get_all_paths(path):
    """ Gets list of data files in path
    Args:
        path (str): data directory
    Returns:
        fls (list): data files
    """
    fls = [os.path.join(path, f) for f in os.listdir(path)]
    return fls

def get_sample_count(path):
    """ Gets number of data files in path
    Args:
        path (str): data directory
    Returns:
        (int): number of data files
    """
    return len(get_all_paths(path))

def get_path_from_index(index, path):
    """ Gets specific data file from path
    Args:
        index (int): specific data file index
        path (str): data directory
    Returns:
        (str): data file
    """
    return get_all_paths(path)[index]

def get_data(index, path):
    """ Loads specific data file
    Args:
        index (int): specific data file index
        path (str): data directory
    Returns:
       (np.array): data from file
    """
    fl = get_path_from_index(index, path)
    return np.load(fl)

def get_correlations(path):
    """ Computes correlation for all files in path
    Args:
        path (str): data directory
    Returns:
        cors (list): correlation between all pairs of properties per file
    """
    N = get_sample_count(path)
    cors = []
    for i in range(N):
        X = get_data(i, path)
        d = X.shape[2]

        C = np.zeros((d,d)) # correlation matrix of properties per file
        for j in range(d):
            for k in range(j, d):
                C[j,k] = C[k,j] = np.corrcoef(X[:,:,j].flatten(), X[:,:,k].flatten())[0][1]

        cors.append(C)

    return cors

def get_correlation_values(cors, r, c):
    """ Gets correlation between properties r and c for all files
    Args:
        cors (list): propertiess correlations per file
        r (int): first property index
        c (int): second property index
    Returns:
        rc_cors (list): correlation between properties r and c per file
    """
    rc_cors = [cor[r,c] for cor in cors]
    return rc_cors

def show_correlations(num_props, path):
    """ Plots correlations between all properties
    Args:
        num_props (int): number of properties
        path (str): data directory
    """
    cors = get_correlations(path)

    fig = pyplot.figure(figsize=(20, 30))
    fig.subplots_adjust(hspace=1)

    cnt = 1
    for i in range(num_props):
        for j in range(num_props):
            pyplot.subplot(num_props, num_props, cnt)
            pyplot.title(f"{PROPS[i]} -- {PROPS[j]}", fontsize=11)

            if i == j: # skip auto-correlation
                pyplot.xlabel("Correlation")
                pyplot.xlim(-1,1)
                pyplot.grid()
                cnt += 1
                continue


            P = get_correlation_values(cors, i, j)
            V = [p for p in P if not np.isnan(p)]

            sb.distplot(V) # TODO warning gets thrown out when using this
            pyplot.xlabel("Correlation")
            pyplot.xlim(-1,1)
            pyplot.grid()

            cnt += 1

    pyplot.show()

## Mixture of Gaussians Model
def segment(X, outliers, num_components=3, normal=True):
    """ Classifies each pixel into components using a Gaussian mixture model
    Args:
        X (np.array): data
        outliers (np.array): outliers mask
        num_components (int): number of classes
        normal (bool): flag to apply normalization of data
    Returns:
        (np.array): matrix of classification per pixel
    """
    n, m, d = X.shape

    if normal: # maps domain to [-1,1]
        max_vector = [np.max(np.abs(X[:,:,i])) for i in range(d)]

        normal_X = np.zeros(X.shape)
        for i in range(d):
            normal_X[:,:,i] = X[:,:,i] / max_vector[i]
    else:
        normal_X = X

    V = [normal_X[i,j,:] for i in range(m) for j in range(n) if not outliers[i,j]]
    V_full = [normal_X[i,j,:] for i in range(n) for j in range(m)]

    gmm = GaussianMixture(n_components=num_components, covariance_type='full')
    gmm.fit(V)

    l = gmm.predict(V_full) # provides a gmm component to all data points
    return l.reshape(n,m)

def apply_segmentation(X, O, height_flag=False):
    """ Gets classification of pixels after segmentation
    Args:
        X (np.array): data
        O (np.array): outliers
        height_flag (bool): flag to keep height property
    Returns:
        L (np.array): matrix of classification per pixel
        reduced_X (np.array): data after applying height flag
    """
    # NOTE keep height data for 2-components data set due to their direct correlation
    if height_flag:
        reduced_X = X
    else: # remove it for other data types to avoid hurting the classification
        reduced_X = np.delete(X, HEIGHT_INDEX, axis=2)

    L = segment(reduced_X, O, normal=True)
    reduced_X[O==1] = 0 # remove outliers from data

    L += 1 # all labels move up one
    L *= (1 - O) # outliers map to label 0

    return L, reduced_X

def show_classification(L, reduced_X):
    """ Shows classification of pixels after segmentation
    Args:
        L (np.array): matrix of classification per pixel
        reduced_X (np.array): data after applying height flag
    """
    d = reduced_X.shape[2]
    fig = pyplot.figure(figsize=(16, 30), dpi= 80, facecolor='w', edgecolor='k')
    cnt = 1
    for i in range(d):
        pyplot.subplot(3,4,cnt)
        pyplot.title(PROPS[i])
        m = pyplot.imshow(reduced_X[:,:,i])
        pyplot.colorbar(m, fraction=0.046, pad=0.04)
        cnt += 1

        pyplot.subplot(3,4,cnt)
        pyplot.title("Segmentation")
        m = pyplot.imshow(L)
        pyplot.colorbar(m, fraction=0.046, pad=0.04)
        cnt += 1

    pyplot.tight_layout()
    pyplot.show()

def show_classification_distributions(L, X):
    """ Shows distributions of classes after segmentation
    Args:
        L (np.array): matrix of classification per pixel
        X (np.array): data
    """
    d = X.shape[2]
    fig = pyplot.figure(figsize=(20, 20), dpi= 80, facecolor='w', edgecolor='k')
    cnt = 1
    for i in range(d):
        pyplot.subplot(3,4,cnt)
        pyplot.title(PROPS[i])
        m = pyplot.imshow(X[:,:,i], aspect='auto')
        pyplot.colorbar(m, fraction=0.046, pad=0.04)
        cnt += 1

        pyplot.subplot(3,4,cnt)
        pyplot.title(PROPS[i])
        for j in range(1, np.max(L) + 1): # skip outliers
            pyplot.hist(X[:,:,i][L == j], 100, alpha=0.3, density=True)

        cnt += 1
        pyplot.grid()

    pyplot.tight_layout()
    pyplot.show()

