import logging

import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import persistence_watershed_algorithm as pws
from methods import main_methods as mm  # NOTE SSTS directory must be in PYTHONPATH

logger = logging.getLogger(__name__)

DEF_THRESH = 0.5
LABEL_THRESH = mm.LABEL_THRESH


class SegmenterWatershed(object):
    def __init__(self, normalize=True, smooth=True):
        """
        Args:
            normalize (bool): Normalize data before processing.
            smooth (bool): Smooth data with neighbor information before processing.
        """
        self.normalize = normalize
        self.smooth = smooth

        self.pws = None

    def fit(self, data):
        """
        Performs Persistence Watershed segmentation on selected material property.

        Args:
            data (NumPy Array): Material property array of shape (height, width).
        """

        if len(data.shape) != 2:
            logger.warning("Data array must be of shape (height, width).")
            return None

        if self.normalize:
            data = SegmenterWatershed.normalize_data(data)

        if self.smooth:
            data = SegmenterWatershed.smooth_data(data)

        self.pws = pws.PersistenceWatershed(data)
        self.pws.train()

    def transform(self, data, outliers=None, pers_thresh=DEF_THRESH):  # NOTE need data as input to use as GMM Segmenter
        """
        Applies threshold to the watershed graph and returns a labelled array.

        Args:
            data (NumPy Array): Material property array of shape (height, width).
            outliers (NumPy Array): Binary array indicating outliers of shape (height, width)
            pers_thresh (float): merging threshold

        Returns:
            (NumPy Array): The segmented array of shape (height, width). Each pixel receives
                a label corresponding to its segment.
        """
        if self.pws is None:
            logger.warning("Attempting to transform prior to fitting. You must call .fit() first.")
            return None

        if self.normalize:
            data = SegmenterWatershed.normalize_data(data)

        if self.smooth:
            data = SegmenterWatershed.smooth_data(data)

        labels = self.pws.apply_threshold(pers_thresh)
        labels = measure.label(labels, connectivity=2, background=0)  # creates unique labels
        if outliers is not None:
            labels *= 1 - outliers  # outliers map to label 0 which are borders between grains

        return labels

    def fit_transform(self, data, outliers=None, pers_thresh=DEF_THRESH):
        """
        Learns and uses persistence watershed graph.

        Args:
            data (Numpy Array): Mateial property array of shape (height, width).
            outliers (NumPy Array): Binary array indicating outliers of shape (height, width)
            pers_thresh (float): merging threshold

        Returns:
            (Numpy Array): The segmented array of shape (height, width). Each pixel receives
                a label corresponding to its segment.
        """
        self.fit(data)
        if self.pws is None:
            logger.warning("Failed to fit model.")
            return None

        return self.transform(data, outliers, pers_thresh)

    @staticmethod
    def normalize_data(data):  # Maps from (-a,b) to (0,1). This allows negative values to be grains.
        return np.abs(data) / np.max(np.abs(data))

    @staticmethod
    def smooth_data(data, window=3):
        center = int(window / 2)
        smoothing_matrix = np.ones((window, window))
        smoothing_matrix[center, center] = 2
        smoothing_matrix /= np.sum(smoothing_matrix)
        return convolve2d(data, smoothing_matrix, mode="same")

    def find_optimal_thresh(self, data, outliers=None, n=50, plot_flag=False):
        """ Finds optimal merging threshold """
        num_grains = []
        thresholds = np.linspace(0, 1, n)
        for thresh in thresholds:
            labels = self.transform(data, outliers, thresh)
            unique_labels = mm.get_unique_labels(labels)
            grain_labels = [l for l in unique_labels if np.sum(labels == l) > LABEL_THRESH]
            num_grains.append(len(grain_labels))

        persistent_value = int(np.median(num_grains))
        optimal_index = num_grains.index(persistent_value)
        optimal_thresh = thresholds[optimal_index]

        if plot_flag:
            plt.figure()
            plt.plot(thresholds, num_grains, "r")
            plt.plot(thresholds, num_grains, "ko")
            plt.grid()
            plt.show()

        return optimal_thresh

