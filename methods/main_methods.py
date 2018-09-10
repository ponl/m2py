import os
import numpy as np
from scipy import signal
from matplotlib import pyplot

from methods import config

fields = config.data_properties

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
    for i in range(6):
        x = X[:,:,i]
        pyplot.subplot(3,4,cnt)
        pyplot.title(fields[i])
        m = pyplot.imshow(x)
        pyplot.colorbar(m, fraction=0.046, pad=0.04)

        cnt += 1

        # Apply Sobel operator
        pyplot.subplot(3,4,cnt)
        pyplot.title(f"{fields[i]} Gradient")
        m = pyplot.imshow(sobel(x))
        pyplot.colorbar(m, fraction=0.046, pad=0.04)

        cnt += 1

    pyplot.tight_layout()
    pyplot.show()

def show_feature_distributions(X):
    """ Plots the pdfs of the data properties
    Args:
        X (np array): data
    """
    O = extract_outliers(X)
    pyplot.figure(figsize=(15, 10), dpi= 80, facecolor='w', edgecolor='k')

    for j in range(6):
        x = [X[n,m,j] for n in range(X.shape[0]) for m in range(X.shape[1]) if not O[n,m]]
        pyplot.subplot(3, 3, j+1)
        pyplot.title(fields[j])
        pyplot.hist(x, 50, density=True)

    pyplot.show()

def extract_outliers(X, height_index = 4, threshold=2.5):
    """ Finds outliers from data
    Args:
        X (np array): data
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

def show_outliers(X):
    """ Plots data properties and outliers
    Args:
        X (np array): data
    """
    fig = pyplot.figure(figsize=(15,5))

    pyplot.subplot(1,3,1)
    m = pyplot.imshow(X[:,:,4])
    pyplot.title('Height')
    pyplot.colorbar(m, fraction=0.046, pad=0.04)

    pyplot.subplot(1,3,2)
    pyplot.imshow(extract_outliers(X))
    pyplot.title('Height')

    outliers = extract_outliers(X)
    X[outliers == 1] = 0
    pyplot.subplot(1,3,3)
    m = pyplot.imshow(X[:,:,4])
    pyplot.title('Height')
    pyplot.colorbar(m, fraction=0.046, pad=0.04)
    pyplot.show()

