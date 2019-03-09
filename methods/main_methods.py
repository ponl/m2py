import os
from collections import Counter

import numpy as np
import seaborn as sb
from scipy import signal
from matplotlib import pyplot, colors, cm
from sklearn.mixture import GaussianMixture

from methods import config

PROPS = config.data_properties
HEIGHT_INDEX = config.height_index
SAMPLE_SIZE = {'backgrounded': 1, '2componentfilms': 0.5, 'nanowires': 5} # nano-meters

LABEL_THRESH = 1000 # each label must have more than this number of pixels

ALPHA = 0.8
NUM_BINS = 30

NUM_COLS = 4 # number of cols in plots

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
    x = data[:, :, height_index]

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
    m = pyplot.imshow(data[:, :, height_index], aspect="auto")
    pyplot.title('Height')
    pyplot.colorbar(m, fraction=0.046, pad=0.04)

    pyplot.subplot(1,3,2)
    pyplot.imshow(outliers, aspect='auto')
    pyplot.title('Height Outliers')

    no_outliers_data = np.copy(data)
    no_outliers_data[outliers == 1] = 0
    pyplot.subplot(1,3,3)
    m = pyplot.imshow(no_outliers_data[:, :, height_index], aspect='auto')
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
                C[j,k] = C[k,j] = np.corrcoef(data[:, :, j].flatten(), data[:, :, k].flatten())[0][1]

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

def get_unique_labels(labels):
    """ Gets unique labels """
    unique_labels = list(np.unique(labels))
    if 0 in unique_labels: # skips outliers AND borders in watershed segmentation
        unique_labels.remove(0)

    unique_labels = sorted(unique_labels, key=lambda k: np.sum(labels==k))
    return unique_labels

def show_classification(labels, data):
    """ Shows classification of pixels after segmentation
    Args:
        labels (np.array): matrix of classification per pixel
        data (np.array): data
    """
    unique_labels = get_unique_labels(labels)
    grain_labels = [l for l in unique_labels if np.sum(labels==l) > LABEL_THRESH]
    num_labels = len(grain_labels)

    h, w, c = data.shape
    num_plots = 2 * c
    num_rows = int(np.ceil(num_plots / NUM_COLS))

    fig = pyplot.figure(figsize=(16, 30), dpi=80, facecolor='w', edgecolor='k')
    cnt = 1
    cmap = pyplot.get_cmap('jet', num_labels)
    for i in range(c):
        ax_l = pyplot.subplot(num_rows, NUM_COLS, cnt)
        cnt += 1
        ax_l.set_title(PROPS[i])
        m = ax_l.imshow(data[:, :, i])
        pyplot.colorbar(m, fraction=0.046, pad=0.04)

        ax_r = pyplot.subplot(num_rows, NUM_COLS, cnt)
        cnt += 1
        ax_r.set_title("Segmentation")
        for index, j in enumerate(grain_labels): # plots mask and distribution per class
            color_step = num_labels - (index + 1)
            mask = np.ma.masked_where(labels!=j, color_step * np.ones(labels.shape))
            ax_r.imshow(mask, alpha=ALPHA, cmap=cmap, vmin=0, vmax=cmap.N)

        colorbar_index(ncolors=num_labels, cmap=cmap)

    pyplot.tight_layout()
    pyplot.show()

def show_classification_distributions(labels, data):
    """ Shows distributions of classes after segmentation
    Args:
        labels (np.array): matrix of classification per pixel
        data (np.array): data
    """
    unique_labels = get_unique_labels(labels)
    grain_labels = [l for l in unique_labels if np.sum(labels==l) > LABEL_THRESH]
    num_labels = len(grain_labels)

    h, w, c = data.shape
    num_plots = 2 * c
    num_rows = int(np.ceil(num_plots / NUM_COLS))

    fig = pyplot.figure(figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
    cnt = 1
    cmap = pyplot.get_cmap('jet', num_labels)
    for i in range(c):
        ax_l = pyplot.subplot(num_rows, NUM_COLS, cnt)
        cnt += 1
        ax_l.set_title(PROPS[i])
        m = ax_l.imshow(data[:, :, i], aspect='auto')
        pyplot.colorbar(m, fraction=0.046, pad=0.04)

        ax_r = pyplot.subplot(num_rows, NUM_COLS, cnt)
        cnt += 1
        ax_r.grid()
        ax_r.set_title(PROPS[i])
        for index, j in enumerate(grain_labels):
            color_step = num_labels - (index + 1)
            ax_r.hist(data[:, :, i][labels == j], NUM_BINS, alpha=ALPHA, density=True, color=cmap(color_step))

    pyplot.tight_layout()
    pyplot.show()

def show_grain_area_distribution(labels, data_type):
    """ Computes a histogram of the number of pixels per label
    Args:
        labels (np.array): matrix of classification per pixel
        data_type (str): type of data such as 2component or backgrounded
    """
    sample_area = SAMPLE_SIZE[data_type] ** 2

    unique_labels = get_unique_labels(labels)
    grain_areas = [np.sum(labels==l) for l in unique_labels]

    sig_areas = sorted([a for a in grain_areas if a > LABEL_THRESH], reverse=True)
    percent_sig_areas = sig_areas / np.sum(grain_areas) * 100
    physical_sig_areas = percent_sig_areas / 100 * sample_area

    pyplot.figure(figsize=(18,5), dpi=80, facecolor='w', edgecolor='k')
    pyplot.subplot(1,3,1)
    pyplot.plot(np.log10(sig_areas), 'ro')
    pyplot.plot(np.log10(sig_areas), 'b')
    pyplot.xlabel('Grain')
    pyplot.ylabel('Log number of pixles per grain')
    pyplot.title('Log Number of Pixels per Grain')
    pyplot.grid()

    pyplot.subplot(1,3,2)
    pyplot.plot(percent_sig_areas, 'ro')
    pyplot.plot(percent_sig_areas, 'b')
    pyplot.xlabel('Grain')
    pyplot.ylabel('Grain percentage (%)')
    pyplot.title('Grain Percentage (%)')
    pyplot.grid()

    pyplot.subplot(1,3,3)
    pyplot.plot(physical_sig_areas, 'ro')
    pyplot.plot(physical_sig_areas, 'b')
    pyplot.xlabel('Grain')
    pyplot.ylabel('Grain area (nm)')
    pyplot.title('Grain Area (nm)')
    pyplot.grid()

    pyplot.show()

def show_distributions_together(labels, data):
    """ Shows distributions of classes after segmentation
    Args:
        labels (np.array): matrix of classification per pixel
        data (np.array): data
    """
    unique_labels = get_unique_labels(labels)
    grain_labels = [l for l in unique_labels if np.sum(labels==l) > LABEL_THRESH]
    num_labels = len(grain_labels)

    h, w, c = data.shape
    num_plots = 2 * c
    num_rows = int(np.ceil(num_plots / NUM_COLS))

    fig = pyplot.figure(figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
    cnt = 1
    cmap = pyplot.get_cmap('jet', num_labels)
    for i in range(c):
        ax_l = pyplot.subplot(num_rows, NUM_COLS, cnt)
        cnt += 1
        ax_l.set_title(PROPS[i])

        ax_r = pyplot.subplot(num_rows, NUM_COLS, cnt)
        cnt += 1
        ax_r.grid()
        ax_r.set_title(PROPS[i])

        for index, j in enumerate(grain_labels): # plots mask and distribution per class
            color_step = num_labels - (index + 1)
            mask = np.ma.masked_where(labels!=j, color_step * np.ones(labels.shape))
            ax_l.imshow(mask, alpha=ALPHA, cmap=cmap, aspect='auto', vmin=0, vmax=num_labels)

            ax_r.hist(data[:, :, i][labels == j], NUM_BINS, alpha=ALPHA, density=True, color=cmap(color_step))

    pyplot.tight_layout()
    pyplot.show()

def show_distributions_separately(labels, data):
    """ Shows distributions of each class separately
    Args:
        labels (np.array): matrix of classification per pixel
        data (np.array): data
    """
    unique_labels = get_unique_labels(labels)
    grain_labels = [l for l in unique_labels if np.sum(labels==l) > LABEL_THRESH]
    num_labels = len(grain_labels)

    h, w, c = data.shape
    num_plots = 2 * c
    num_rows = int(np.ceil(num_plots / NUM_COLS))

    cmap = pyplot.get_cmap('jet', num_labels)
    for index, gl in enumerate(grain_labels):
        color_step = num_labels - (index + 1)
        fig = pyplot.figure(figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
        cnt = 1
        for i in range(c):
            ax_l = pyplot.subplot(num_rows, NUM_COLS, cnt)
            cnt += 1
            ax_l.set_title(PROPS[i])
            m = ax_l.imshow(data[:, :, i], aspect='auto')
            mask = np.ma.masked_where(labels==gl, np.ones(labels.shape))
            ax_l.imshow(mask, alpha=1, cmap='bone', aspect='auto', vmin=0, vmax=1)
            pyplot.colorbar(m, fraction=0.046, pad=0.04)

            ax_r = pyplot.subplot(num_rows, NUM_COLS, cnt)
            cnt += 1
            ax_r.set_title(PROPS[i])
            ax_r.hist(data[:, :, i][labels == gl], NUM_BINS, alpha=ALPHA, density=True, color=cmap(color_step))
            ax_r.grid()

        pyplot.tight_layout()
        pyplot.show()

def show_overlaid_distribution(probs, data):
    """ Plots distributions overlaid on pixels
    Args:
        probs (np.array): array of probabilities per pixel
        data (np.array): data
    """
    h, w, n = probs.shape
    h, w, c = data.shape

    num_plots = n + c
    num_rows = int(np.ceil(num_plots / NUM_COLS))

    fig = pyplot.figure(figsize=(20, 20), dpi=80, facecolor='w', edgecolor='k')
    cnt = 1
    for i in range(c):
        ax = pyplot.subplot(num_rows, NUM_COLS, cnt)
        cnt += 1
        m = ax.imshow(data[:, :, i])
        pyplot.colorbar(m, fraction=0.046, pad=0.04)
        ax.set_title(PROPS[i])

    for i in range(n):
        ax = pyplot.subplot(num_rows, NUM_COLS, cnt)
        cnt += 1
        m = ax.imshow(probs[:, :, i], vmin=0, vmax=1)
        pyplot.colorbar(m, fraction=0.046, pad=0.04)

    pyplot.show()

# NOTE Auxilliary methods for creating discrete colorbar
def colorbar_index(ncolors, cmap):
    """ Adds discrete colorbar to plot.
    Args:
        ncolors: number of colors.
        cmap: colormap instance
    """
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = pyplot.colorbar(mappable, fraction=0.046, pad=0.04)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
    Args:
        cmap: colormap instance.
        N: number of colors.
    """
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki, key in enumerate(('red','green','blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1, ki], colors_rgba[i, ki]) for i in range(N+1)]

    return colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict)


