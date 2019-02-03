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
MAX_PIXEL = 255
CMAP = pyplot.get_cmap('gist_ncar')

## Boundary Detection
def sobel(data):
    """ Applies the Sobel operator to x
    Args:
        data (np array): data
    Returns:
        (np array): data after applying convolution
    """
    fr = np.array([[0, 1, 0, -1, 0], [1, 2, 0, -2, -1], [2, 4, 0, -4, 2], [ 1,  2,  0, -2,  1], [0,  1,  0, -1, 0]])
    fc = np.array([[0, 1, 2,  1, 0], [1, 2, 4,  2,  1], [0, 0, 0,  0, 0], [-1, -2, -4, -2, -1], [0, -1, -2, -1, 0]])
    yr = signal.convolve2d(data, fr, boundary="symm", mode="same")
    yc = signal.convolve2d(data, fc, boundary="symm", mode="same")

    return np.sqrt(yr**2 + yc**2)

def show_boundaries(data):
    """ Plots the data properties and their boundaries
    Args:
        data (np array): data
    """
    fig = pyplot.figure(figsize=(16, 30), dpi=80, facecolor='w', edgecolor='k')
    sobel_data = np.zeros(data.shape)
    cnt = 1
    for i in range(data.shape[2]):
        x = data[:,:,i]
        pyplot.subplot(3,4,cnt) # TODO how to ensure break down of plots?
        pyplot.title(PROPS[i])
        m = pyplot.imshow(x)
        pyplot.colorbar(m, fraction=0.046, pad=0.04)

        cnt += 1

        # Apply Sobel operator
        x_sobel = sobel(x)
        sobel_data[:,:,i] = x_sobel

        pyplot.subplot(3,4,cnt)
        pyplot.title(f"{PROPS[i]} Gradient")
        m = pyplot.imshow(x_sobel)
        pyplot.colorbar(m, fraction=0.046, pad=0.04)

        cnt += 1

    pyplot.tight_layout()
    pyplot.show()

    return sobel_data

## Properties Distributions
def show_property_distributions(data, outliers):
    """ Plots the pdfs of the data properties
    Args:
        data (np array): data
        outliers (np array): outliers
    """
    fig = pyplot.figure(figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')

    h, w, c = data.shape
    for j in range(c):
        x = [data[n,m,j] for n in range(h) for m in range(w) if not outliers[n,m]]
        pyplot.subplot(2, 3, j+1) # TODO how to ensure break down of plots?
        pyplot.title(PROPS[j])
        sb.distplot(x) # TODO warning gets thrown out when using this
        pyplot.grid()

    pyplot.show()

## Outlier Detection
def extract_outliers(data, height_index=HEIGHT_INDEX, threshold=2.5):
    """ Finds outliers from data
    Args:
        data (np array): data
        height_index (int): index of height property
        threshold: (float): z-score threshold
    Returns:
        (np array): boolean matrix denoting outliers
    """
    x = data[:,:,height_index]

    # Smooth data
    flt = np.array([[0.5, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 0.5]])
    y = signal.convolve2d(x, flt, boundary='symm', mode='same')

    # Compute z-scores
    u = np.mean(y)
    s = np.std(y)
    z = np.abs((u-y)/s)

    # Threshold by z-score
    return z > threshold

def show_outliers(data, outliers, height_index=HEIGHT_INDEX):
    """ Plots data properties and outliers
    Args:
        data (np array): data
        outliers (np array): outliers
        height_index (int): index of height property
    """
    fig = pyplot.figure(figsize=(18,5))

    pyplot.subplot(1,3,1)
    m = pyplot.imshow(data[:,:,height_index], aspect="auto")
    pyplot.title('Height')
    pyplot.colorbar(m, fraction=0.046, pad=0.04)

    pyplot.subplot(1,3,2)
    pyplot.imshow(outliers, aspect='auto')
    pyplot.title('Height Outliers')

    no_outliers_data = np.copy(data)
    no_outliers_data[outliers == 1] = 0
    pyplot.subplot(1,3,3)
    m = pyplot.imshow(no_outliers_data[:,:,height_index], aspect='auto')
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
        data = get_data(i, path)
        c = data.shape[2]

        C = np.zeros((c,c)) # correlation matrix of properties per file
        for j in range(c):
            for k in range(j, c):
                C[j,k] = C[k,j] = np.corrcoef(data[:,:,j].flatten(), data[:,:,k].flatten())[0][1]

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
def segment(data, outliers, num_components=2):
    """ Classifies each pixel into components using a Gaussian mixture model
    Args:
        data (np.array): data
        outliers (np.array): outliers mask
        num_components (int): number of classes
        normal (bool): flag to apply normalization of data
    Returns:
        (np.array): matrix of classification per pixel
    """
    h, w, c = data.shape
    n = h * w

    # Normalize full data for prediction
    data = data.reshape(n, c)
    m = np.max(np.abs(data), axis=0)
    normal_data = data / m

    # Normalize data without outliers for fitting
    outliers = outliers.flatten()
    no_outliers_data = [data[i] for i in range(n) if not outliers[i]]
    m = np.max(np.abs(no_outliers_data), axis=0)
    normal_no_outliers_data = no_outliers_data / m

    gmm = GaussianMixture(n_components=num_components, covariance_type='full')
    gmm.fit(normal_no_outliers_data)

    l = gmm.predict(normal_data) # provides a gmm component to all data points
    return l.reshape(h, w)

def apply_segmentation(data, outliers):
    """ Gets classification of pixels after segmentation
    Args:
        data (np.array): data
        outliers (np.array): outliers
    Returns:
        labels (np.array): matrix of classification per pixel
    """
    labels = segment(data, outliers)

    labels += 1 # all labels move up one
    labels *= (1 - outliers) # outliers map to label 0

    return labels

def get_unique_labels(labels):
    """ Gets unique labels """
    unique_labels = list(np.unique(labels))
    if 0 in unique_labels: # skips outliers AND borders in watershed segmentation
        unique_labels.remove(0)

    return unique_labels

def show_classification(labels, data):
    """ Shows classification of pixels after segmentation
    Args:
        labels (np.array): matrix of classification per pixel
        data (np.array): data
    """
    c = data.shape[2]
    fig = pyplot.figure(figsize=(16, 30), dpi=80, facecolor='w', edgecolor='k')
    cnt = 1
    for i in range(c):
        pyplot.subplot(3,4,cnt)
        pyplot.title(PROPS[i])
        m = pyplot.imshow(data[:,:,i])
        pyplot.colorbar(m, fraction=0.046, pad=0.04)
        cnt += 1

        pyplot.subplot(3,4,cnt)
        pyplot.title("Segmentation")
        m = pyplot.imshow(labels, cmap=CMAP)
        pyplot.colorbar(m, fraction=0.046, pad=0.04)
        cnt += 1

    pyplot.tight_layout()
    pyplot.show()

def show_classification_distributions(labels, data):
    """ Shows distributions of classes after segmentation
    Args:
        labels (np.array): matrix of classification per pixel
        data (np.array): data
    """
    unique_labels = get_unique_labels(labels)

    c = data.shape[2]
    fig = pyplot.figure(figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
    cnt = 1
    for i in range(c):
        ax_l = pyplot.subplot(3,4,cnt)
        cnt += 1
        ax_l.set_title(PROPS[i])
        m = pyplot.imshow(data[:,:,i], aspect='auto')
        pyplot.colorbar(m, fraction=0.046, pad=0.04)

        ax_r = pyplot.subplot(3,4,cnt)
        cnt += 1
        ax_r.grid()
        ax_r.set_title(PROPS[i])
        for index, j in enumerate(unique_labels):
            color_step = int((index + 1) * MAX_PIXEL / len(unique_labels))
            ax_r.hist(data[:,:,i][labels == j], 50, alpha=0.8, density=True, color=CMAP(MAX_PIXEL - color_step))

    pyplot.tight_layout()
    pyplot.show()

def show_grain_area_distribution(labels):
    """ Computes a histogram of the number of pixels per label
    Args:
        labels (np.array): matrix of classification per pixel
    """
    unique_labels = get_unique_labels(labels)
    grain_areas = [np.sum(labels==l) for l in unique_labels]
    normal_grain_areas = grain_areas / np.sum(grain_areas) * 100

    pyplot.figure(figsize=(18,5), dpi=80, facecolor='w', edgecolor='k')
    pyplot.subplot(1,3,1)
    pyplot.plot(unique_labels, grain_areas, 'ro')
    pyplot.plot(unique_labels, grain_areas, 'b')
    pyplot.xlabel('Label numbering')
    pyplot.ylabel('Grain area')
    pyplot.title('Grain Area (number of pixels)')
    pyplot.grid()

    pyplot.subplot(1,3,2)
    pyplot.plot(unique_labels, np.log10(grain_areas), 'ro')
    pyplot.plot(unique_labels, np.log10(grain_areas), 'b')
    pyplot.xlabel('Label numbering')
    pyplot.ylabel('Log grain area')
    pyplot.title('Log Grain Area (number of pixels)')
    pyplot.grid()

    pyplot.subplot(1,3,3)
    pyplot.plot(unique_labels, normal_grain_areas, 'ro')
    pyplot.plot(unique_labels, normal_grain_areas, 'b')
    pyplot.xlabel('Label numbering')
    pyplot.ylabel('Percentage in image')
    pyplot.title('Grain Percentage in Image')
    pyplot.grid()

    pyplot.show()


def show_distributions_together(labels, data):
    """ Shows distributions of classes after segmentation
    Args:
        labels (np.array): matrix of classification per pixel
        data (np.array): data
    """
    unique_labels = get_unique_labels(labels)
    grain_labels = [l for l in unique_labels if np.sum(labels==l) > 1000]

    c = data.shape[2]
    fig = pyplot.figure(figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
    cnt = 1
    for i in range(c):
        ax_l = pyplot.subplot(3,4,cnt)
        cnt += 1
        ax_l.set_title(PROPS[i])
        m = ax_l.imshow(data[:,:,i], aspect='auto')

        ax_r = pyplot.subplot(3,4,cnt)
        cnt += 1
        ax_r.grid()
        ax_r.set_title(PROPS[i])
        for index, j in enumerate(grain_labels): # plots mask and distribution per class
            color_step = int((index + 1) * MAX_PIXEL / len(grain_labels))
            mask = np.ma.masked_where(labels!=j, MAX_PIXEL * np.ones(labels.shape) - color_step)
            ax_l.imshow(mask, alpha=0.8, cmap=CMAP, aspect='auto', vmin=0, vmax=MAX_PIXEL)

            ax_r.hist(data[:,:,i][labels == j], 50, alpha=0.8, density=True, color=CMAP(MAX_PIXEL - color_step))

    pyplot.tight_layout()
    pyplot.show()

def show_distributions_separately(labels, data):
    """ Shows distributions of each class separately
    Args:
        labels (np.array): matrix of classification per pixel
        data (np.array): data
    """
    unique_labels = get_unique_labels(labels)
    grain_labels = [l for l in unique_labels if np.sum(labels==l) > 1000]

    for gl in grain_labels:
        c = data.shape[2]
        fig = pyplot.figure(figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
        cnt = 1
        for i in range(c):
            ax_l = pyplot.subplot(3,4,cnt)
            cnt += 1
            ax_l.set_title(PROPS[i])
            m = ax_l.imshow(data[:,:,i], aspect='auto')
            mask = np.ma.masked_where(labels==gl, np.ones(labels.shape))
            ax_l.imshow(mask, alpha=1, cmap='bone', aspect='auto', vmin=0, vmax=1)
            pyplot.colorbar(m, fraction=0.046, pad=0.04)

            ax_r = pyplot.subplot(3,4,cnt)
            cnt += 1
            ax_r.set_title(PROPS[i])
            ax_r.hist(data[:,:,i][labels == gl], 50, alpha=0.6, density=True, color='k')
            ax_r.grid()

        pyplot.tight_layout()
        pyplot.show()

