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
    """For a specified  pixel (i, j), we get its closest (four neighbors) non-outlier value from labels

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
    if zeros is None:
        return labels

    if len(labels.shape) < 3:
        labels = np.expand_dims(labels, axis=2)

    c = labels.shape[2]

    x, y = np.nonzero(zeros)
    for i, j in zip(x, y):
        for k in range(c):
            value = get_closest_value(labels[:, :, k], zeros, i, j)
            labels[i, j, k] = value

    return np.squeeze(labels)


def get_significant_labels(labels, label_thresh=10, bg_thresh=None):
    """
    Shows classification of pixels after segmentation

    Parameters
    ----------
        labels : NumPy Array
            matrix of classification per pixel
        label_thresh : int
            removes grains that are smaller than this value
        bg_thresh : int
            highlights grains (background grains) that are bigger than this value

    Returns
    ----------
        new_labels : NumPy Array
            matrix of classification per pixel for large components
    """
    unique_labels = get_unique_labels(labels)
    grain_labels = [l for l in unique_labels if np.sum(labels == l) > label_thresh]
    num_labels = len(grain_labels)

    if bg_thresh:
        bg_labels = [l for l in grain_labels if np.sum(labels == l) > bg_thresh]
    else:
        bg_labels = []

    h, w = labels.shape
    new_labels = np.zeros((h, w))
    for index, j in enumerate(grain_labels):  # plots mask per class
        if j in bg_labels:
            color_step = num_labels  # uses a distinct color
        else:
            color_step = num_labels - index

        new_labels[labels == j] += color_step

    if not bg_thresh:
        new_labels = relabel(new_labels)

    return new_labels
