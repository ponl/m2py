import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from m2py.utils import config

"""
This module contains functions for sorting and grouping labels resulting
from the GMM segmentation and clustering workflows. Labels and their
descriptive statistics are dynamically sorted into dictionaries of lists
and arrays so that they may be iterably accessed and analyzed.
"""

LABEL_THRESH = 10  # each label must have more than this number of pixels
BG_THRESH = 100000 # NOTE 10k for smaller grains and 100k for bigger grains

data_channels = config.data_info["QNM"]["properties"]

def relabel(labels):
    """
    Relabel labels in order

    Parameters
    ----------
        labels : NumPy Array
            matrix of classification per pixel

    Returns
    ----------
        labels : NumPy Array
            matrix of classification per pixel in order
    """
    unique_labels = get_unique_labels(labels)[::-1]
    max_label = max(unique_labels)
    for i, l in enumerate(unique_labels):
        labels[labels == l] = max_label + i + 1

    for l in get_unique_labels(labels):
        labels[labels == l] -= max_label

    return labels


def get_unique_labels(labels):
    """
    Gets unique labels

    Parameters
    ----------
        labels : NumPy Array
            matrix of classification per pixel

    Returns
    ----------
        unique_labels : NumPy Array
            1D array that lists all of the unique labels
    """
    labels = labels.astype(np.int64)
    unique_labels = [a for a in np.unique(labels) if isinstance(a, np.int64)]
    if 0 in unique_labels:  # skips outliers AND borders in watershed segmentation
        unique_labels.remove(0)

    unique_labels = sorted(unique_labels, key=lambda k: np.sum(labels == k))
    return unique_labels


def get_closest_value(labels, outliers, i, j):
    """ For a specified  pixel (i, j), we get its closest (four neighbors) non-outlier value from labels
    
    Parameters
    ----------
        labels : NumPy Array
            matrix of classification per pixel
        outliers : NumPy Array
            outliers
        i : int
            row value of pixel
        j : int
            col value of pixel
            
    Returns
    ----------
        value : int
            closest non-outlier value.
    """
    h, w = labels.shape

    distance = np.Inf

    k = min(i + 1, h - 1)
    while True:
        if outliers[k, j]:
            k += 1
            if k == h:
                break
        else:
            value = labels[k, j]
            distance = k - i
            break

    k = max(i - 1, 0)
    while True:
        if outliers[k, j]:
            k -= 1
            if k == -1:
                break
        else:
            if i - k < distance:
                distance = i - k
                value = labels[k, j]

            break

    k = min(j + 1, w - 1)
    while True:
        if outliers[i, k]:
            k += 1
            if k == w:
                break
        else:
            if k - j < distance:
                distance = k - j
                value = labels[i, k]

            break

    k = max(j - 1, 0)
    while True:
        if outliers[i, k]:
            k -= 1
            if k == -1:
                break
        else:
            if j - k < distance:
                distance = j - k
                value = labels[i, k]

            break

    return value


def fill_out_zeros(labels, zeros):
    """
    For each zero value, we replace its value in labels by its closest non-zero value
    
    Parameters
    ----------
        labels : NumPy Array
            matrix of classification per pixel
        zeros : NumPy Array
            zero values
    Returns
    ----------
        labels : int
            closest non-outlier value.
    """
    if len(labels.shape) < 3:
        labels = np.expand_dims(labels, axis=2)

    c = labels.shape[2]
    x, y = np.nonzero(zeros)
    for i, j in zip(x, y):
        for k in range(c):
            value = get_closest_value(labels[:,:,k], zeros, i, j)
            labels[i, j, k] = value

    return np.squeeze(labels)


def get_significant_labels(labels, bg_contrast_flag=False, label_thresh=LABEL_THRESH):
    """
    Shows classification of pixels after segmentation
    
    Parameters
    ----------
        labels : NumPy Array
            matrix of classification per pixel
        bg_contrast_flag : bool
            highlights biggest grain (background) in plot
            
    Returns
    ----------
        new_labels : NumPy Array
            matrix of classification per pixel for large components
    """
    unique_labels = get_unique_labels(labels)
    grain_labels = [l for l in unique_labels if np.sum(labels == l) > label_thresh]
    bg_labels = [l for l in grain_labels if np.sum(labels == l) > BG_THRESH]
    num_labels = len(grain_labels)

    h, w = labels.shape
    new_labels = np.zeros((h, w))
    for index, j in enumerate(grain_labels):  # plots mask per class
        if (j in bg_labels) and bg_contrast_flag:
            color_step = num_labels  # uses a distinct color
        else:
            color_step = num_labels - index

        new_labels[labels == j] += color_step

    if not bg_contrast_flag:
        new_labels = relabel(new_labels)

    return new_labels


def array_stats(array):
    """
    Takes in numpy array of data or labels and calculates median, standard 
    deviation, and variance

    Parameters
    ----------
        array : NumPy Array
            single channel of data array

    Returns
    ----------
        median : float64
            median value of the array
        std_dev : float64
            standard deviation of the array
        var : float64
            variance of the array
    """
    median = np.median(array)
    std_dev = np.std(array)
    var = np.var(array)

    return median, std_dev, var


def phase_sort(array, labels, n_components):
    """
    Takes in a 3D numpy array and sorts pixels into a dictionary based on phase
    labels
    
    Parameters
    ----------
        array : NumPy Array
            Array of SPM data
        phase_labels : NumPy Array
            array of phase labels
        n_components : int
            number of phases for sorting
    
    Returns
    ----------
        phases : dict
            Dictionary of numpy arrays, where each key is the phase number and 
            each value is a flattened array of the pixels in that phase and 
            their properties
    """
    x, y, z = array.shape

    data = np.reshape(array, ((x * y), z))
    labels = np.reshape(labels, (x * y))

    phase_list = get_unique_labels(labels)
    phase_list.sort()

    phase0 = []
    phase1 = []
    phase2 = []
    phase3 = []
    phase4 = []
    phase5 = []
    phase6 = []
    phase7 = []
    phase8 = []
    phase9 = []

    for i in range(len(labels)):
        if labels[i] == 0:
            phase0.append(data[i, :])
        elif labels[i] == 1:
            phase1.append(data[i, :])
        elif labels[i] == 2:
            phase2.append(data[i, :])
        elif labels[i] == 3:
            phase3.append(data[i, :])
        elif labels[i] == 4:
            phase4.append(data[i, :])
        elif labels[i] == 5:
            phase5.append(data[i, :])
        elif labels[i] == 6:
            phase6.append(data[i, :])
        elif labels[i] == 7:
            phase7.append(data[i, :])
        elif labels[i] == 8:
            phase8.append(data[i, :])
        elif labels[i] == 9:
            phase9.append(data[i, :])
        else:
            pass

    phases = {}

    if len(phase0) >= 1:
        phases[0] = np.asarray(phase0)
    else:
        pass

    if len(phase1) >= 1:
        phases[1] = np.asarray(phase1)
    else:
        pass

    if len(phase2) >= 1:
        phases[2] = np.asarray(phase2)
    else:
        pass

    if len(phase3) >= 1:
        phases[3] = np.asarray(phase3)
    else:
        pass

    if len(phase4) >= 1:
        phases[4] = np.asarray(phase4)
    else:
        pass

    if len(phase5) >= 1:
        phases[5] = np.asarray(phase5)
    else:
        pass

    if len(phase6) >= 1:
        phases[6] = np.asarray(phase6)
    else:
        pass

    if len(phase7) >= 1:
        phases[7] = np.asarray(phase7)
    else:
        pass

    if len(phase8) >= 1:
        phases[8] = np.asarray(phase8)
    else:
        pass

    if len(phase9) >= 1:
        phases[9] = np.asarray(phase9)
    else:
        pass

    return phases


def plot_single_phase_props(array):
    """
    Takes in a flattened array of SPM data and plots the histogram distributions
    of a single phase of the data
    
    Parameters
    ----------
        array : NumPy Array
            Array of SPM data
    
    Returns
    ----------
        
    """
    xy, z = array.shape
    print("Shape: ", array.shape)

    fig = plt.figure(figsize=(10, 15))
    for i in range(z):
        counts, bins = np.histogram(array[:, i], bins=30)
        ymax = counts.max()

        median, std_dev, var = array_stats(array[:, i])

        plt.subplot(3, 2, 1 + i)
        plt.hist(array[:, i], bins=30, alpha=0.5)
        plt.plot([median, median], [0, ymax + 5], label="Median", linewidth=3, c="r")
        plt.plot([median + std_dev, median + std_dev], [0, ymax + 5], label="Std Dev", linewidth=2, c="k")
        plt.plot([median - std_dev, median - std_dev], [0, ymax + 5], linewidth=2, c="k")
        plt.title(
            f"{data_channels[i]}" + "  Median: " + "{:.2e}".format(median) + ",Stand. Dev.: " + "{:.2e}".format(std_dev)
        )
        plt.ylim(0, ymax + 5)
        plt.legend()

    plt.tight_layout()
    plt.show()

    return


def plot_all_phases_props(phases):
    """
    Takes in a flattened array of SPM data and plots the histogram distributions
    of all phases of the data
    
    Parameters
    ----------
        phases : dict
            Dictionary of numpy arrays, where each key is the phase number and 
            each value is a flattened array of the pixels in that phase and 
            their properties
    
    Returns
    ----------
        
    """
    for k, v in phases.items():
        print("Phase ", k)
        plot_single_phase_props(v)

    return


def gen_phase_stats(phases):
    """
    Takes in dictionary of phases from phase_sort() and generates a parallel 
    dictionary of stats
    
    Parameters
    ----------
        phases : dict
            Dictionary of numpy arrays, where each key is the phase number and 
            each value a flattened array of the
            pixels in that phase and their properties
        
    Returns
    ----------
        phase_stats : dict
            Dictionary of numpy arrays, where each key is the phase number and 
            each value is a flattened array of the basic statistical analysis of
            the phase properties
    """
    phase_stats = {}
    keys = ["median", "standard deviation", "variance"]

    for k, v in phases.items():
        xy, z = v.shape
        for i in range(z):
            phase_stats[k] = {}
        for h in range(z):
            phase_stats[k][h] = array_stats(v[:, h])

    return phase_stats


def grain_sort(array, grain_labels):
    """
    Takes in an array and the associated domain (grain) labels. Sorts pixels 
    into a dictionary, where each value is an array of the pixel values in 
    the domain
    
    Parameters
    ----------
        array : NumPy Array
            Array of SPM data
        domain_labels : NumPy Array
            Array of domain labels
        n_components : int
            number of phases for sorting
    
    Returns
    ----------
        grain_props : dict
            Dictionary of numpy arrays, where each key is the phase number and
            each value is a flattened array of the pixels in that grain and 
            their properties
    """
    x, y, z = array.shape

    data = np.reshape(array, ((x * y), z))
    labels = np.reshape(grain_labels, (x * y))

    unique_labels = get_unique_labels(labels)
    unique_labels.sort()

    grain_count = len(unique_labels)

    grain_props = {}
    for h in range(x * y):
        pixel_props = data[h, :]
        label = labels[h]

        if label in grain_props:
            old_value = np.asarray(grain_props[label])

            # Append new pixel vector to old vectors
            new_value = np.vstack((old_value, pixel_props))
        else:
            new_value = np.asarray(pixel_props)

        grain_props[label] = new_value

    return grain_props


def gen_grain_stats(grain_props):
    """
    Takes in a dictionary of the different grains, produced by grain_sort(),
    and calculates basic histogram
    statistics on the constituent pixels
    
    Parameters
    ----------
        grain_props : dict
            Dictionary of numpy arrays, where each key is the phase number and
            each value is a flattened array of the pixels in that grain and 
            their properties
        
    Returns
    ----------
        grain_stats : dict
            Dictionary of numpy arrays, where each key is the grain number and 
            each value is a flattened array of the basic statistical analysis 
            of the grain properties
    """
    grain_stats = {}
    keys = ["median", "standard deviation", "variance"]

    for k, v in grain_props.items():
        grain = np.asarray(v)

        if len(grain.shape) == 1:
            tup = grain.shape
            z = tup[0]
            for i in range(z):
                grain_stats[k] = {}
            for i in range(z):
                grain_stats[k][i] = array_stats(grain[i])
        else:
            xy, z = grain.shape
            for i in range(z):
                grain_stats[k] = {}

            for i in range(z):
                grain_stats[k][i] = array_stats(grain[:, i])

    return grain_stats
