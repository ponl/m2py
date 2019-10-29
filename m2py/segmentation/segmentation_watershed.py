import logging

import numpy as np
from scipy.signal import convolve2d
import m2py.segmentation.persistence_watershed_algorithm as pws
from m2py.utils import seg_label_utils as slu

logger = logging.getLogger(__name__)

DEF_THRESH = 0.5


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

    def fit(self, data, pers_thresh=DEF_THRESH):
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
        self.pws.train(pers_thresh)

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
        if outliers is not None:
            labels *= 1 - outliers  # outliers map to label 0 which are borders between grains

        labels = slu.relabel(labels)
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
        self.fit(data, pers_thresh)
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

