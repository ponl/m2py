import logging
from collections import Counter

import numpy as np
from skimage import measure
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from m2py.utils import seg_label_utils as slu


logger = logging.getLogger(__name__)


class SegmenterGMM(object):
    def __init__(self, n_components=3, normalize=True, embedding_dim=None, padding=0, zscale=False, nonlinear=False):
        """
        Perform material segmentation using a Gaussian Mixture Model. Various ways of building features
        and performing dimensionality reduction are provided. If `padding` is set greater than zero, local
        neighborhoods around each pixel will be used. These neighborhoods will be flattened into vectors
        and fed through the dimensionality reduction step. If `padding` is zero, no neighborhood information
        will be used. The `embedding_dim` value will determine whether to use dimensionality reduction and,
        if so, what the embedded dimension should be. The dimensionality step will be performed prior to
        fitting the Gaussian mixture model.

        Args:
            n_components (int): Number of components (i.e. segments) to learn in the GMM.
            normalize (bool): Normalize the material properties before processing.
            embedding_dim (int): Dimension used in dimensionality reduction.
            padding (int): Padding used in sliding window. When set to zero, no neighbor information
                will be used. The neighbor values are flattened and fed into the PCA routine.
            zscale (bool): Whether to use z-score scaling (True) or scale by property maxima (False).
            nonlinear (bool): Whether to add nonlinear features (True) or not (False).
        """
        self.normalize = normalize
        self.embedding_dim = embedding_dim
        self.n_components = n_components
        self.padding = padding
        self.zscale = zscale
        self.nonlinear = nonlinear

        self.gmm = None
        self.pca = None
        self.sst = None

    def fit(self, data, outliers=None):
        """
        Performs Gaussian mixture model segmentation using
        the provided settings.

        Args:
            data (NumPy Array): Material properties array of shape (height, width, n_properties)
            outliers (NumPy Array): Binary array indicating outliers of shape (height, width)

        Returns:
            self (SegmenterGMM)
        """
        if len(data.shape) != 3:
            logger.warning("Data arrays must be of shape (height, width, n_properties).")
            return None

        if outliers is not None and len(outliers.shape) != 2:
            logger.warning("Outlier arrays must be of shape (height, width).")
            return None

        if self.nonlinear:
            data = SegmenterGMM.add_nonlinear_features(data)

        h, w, c = data.shape
        n = h * w

        if self.padding > 0:
            data = SegmenterGMM.get_windows(data, padding=self.padding)
        else:
            data = data.reshape(n, c)

        if outliers is not None:
            outliers = outliers.flatten()
            data = data[outliers == 0]

        if self.normalize:
            if self.zscale:
                self.sst = StandardScaler()
                data = self.sst.fit_transform(data)
            else:
                data = SegmenterGMM.normalize_by_max(data)

        if self.embedding_dim is not None:
            self.pca = PCA(n_components=self.embedding_dim)
            data = self.pca.fit_transform(data)

        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type="full")
        self.gmm.fit(data)
        return self

    def transform(self, data, outliers=None):
        """
        Applies the learned Gaussian mixture model and
        returns a labelled array.

        Args:
            data (NumPy Array): Material properties array of shape (height, width, n_properties)
            outliers (NumPy Array): Binary array indicating outliers of shape (height, width)

        Returns:
            labels (NumPy Array): The segmented array of shape (height, width). Each pixel receives
                a label corresponding to its segment.
        """
        if self.gmm is None:
            logger.warning("Attempting to transform prior to fitting. You must call .fit() first.")
            return None

        h, w, c = data.shape
        data = self.get_pca_components(data)

        labels = self.gmm.predict(data)
        labels = np.reshape(labels, (h, w))
        labels += 1  # all labels move up one
        if outliers is not None:
            labels *= 1 - outliers  # outliers map to label 0

        labels = slu.relabel(labels)
        return labels

    def fit_transform(self, data, outliers=None):
        """
        Learns and applies a learned Gaussian mixture model and
        returns a labelled array.

        Args:
            data (NumPy Array): Material properties array of shape (height, width, n_properties)
            outliers (NumPy Array): Binary array indicating outliers of shape (height, width)

        Returns:
            labels (NumPy Array): The segmented array of shape (height, width). Each pixel receives
                a label corresponding to its segment.
        """
        gmm = self.fit(data, outliers)
        if gmm is None:
            return None

        return self.transform(data, outliers)

    def get_pca_components(self, data):
        """
        Gathers PCA components.

        Args:
            data (NumPy Array): Material properties array of shape (height, width, n_properties)

        Returns:
            pca_components (NumPy Array): PCA components array of shape (height * width, embedding_dim)
        """
        if self.gmm is None:
            logger.warning("Attempting to access model prior to fitting. You must call .fit() first.")
            return None

        if self.nonlinear:
            data = SegmenterGMM.add_nonlinear_features(data)

        h, w, c = data.shape
        n = h * w

        if self.padding > 0:
            data = SegmenterGMM.get_windows(data, padding=self.padding)
        else:
            data = data.reshape(n, c)

        if self.normalize:
            data = self.sst.transform(data) if self.zscale else SegmenterGMM.normalize_by_max(data)

        if self.embedding_dim is not None:
            data = self.pca.transform(data)

        pca_components = data
        return pca_components

    def store_pca_components(self, data, output_file):
        """
        Stores PCA components.

        Args:
            data (NumPy Array): Material properties array of shape (height, width, n_properties)
            output_file (str): Output file for PCA components
        """
        if self.gmm is None:
            logger.warning("Attempting to access model prior to fitting. You must call .fit() first.")
            return None

        h, w, c = data.shape
        pca_components = self.get_pca_components(data)
        pca_components = pca_components.reshape(h, w, self.embedding_dim)
        np.save(output_file, pca_components)

    def get_probabilities(self, data, outliers=None):
        """
        Computes likelihood of pixel for all classes.

        Args:
            data (NumPy Array): Material properties array of shape (height, width, n_properties)
            outliers (NumPy Array): Binary array indicating outliers of shape (height, width)

        Returns:
            probs (NumPy Array): The array of probabilities of shape (height, width, num_components).
                Each pixel receives a probability per class.
        """
        if self.gmm is None:
            logger.warning("Attempting to access model prior to fitting. You must call .fit() first.")
            return None

        h, w, c = data.shape
        data = self.get_pca_components(data)

        vector_probs = self.gmm.predict_proba(data)
        probs = vector_probs.reshape(h, w, self.n_components)
        return probs

    @staticmethod
    def add_nonlinear_features(data):
        abs_data = np.abs(data)
        squared_data = data ** 2
        cubed_data = data ** 3
        reciprocal_data = 1 / data
        reciprocal_data[data == 0] = 0
        return np.concatenate((data, abs_data, squared_data, cubed_data, reciprocal_data), axis=2)

    @staticmethod
    def get_windows(data, padding=3, flatten=True):
        h, w, c = data.shape
        n = 2 * padding + 1
        wins = np.zeros((h * w, n ** 2, c))
        for d in range(c):
            X = np.pad(data[:, :, d], padding, "symmetric")
            idx = 0
            for i in range(padding, h + padding):
                for j in range(padding, w + padding):
                    imin, imax = i - padding, i + padding + 1
                    jmin, jmax = j - padding, j + padding + 1
                    wins[idx, :, d] = X[imin:imax, jmin:jmax].flatten()
                    idx += 1

        if flatten:
            l = wins.shape[1]
            wins = np.reshape(wins, (-1, l * c))  # (pixels) x (properties + neighbors)

        return wins

    @staticmethod
    def normalize_by_max(data):
        m = np.max(np.abs(data), axis=0)
        return data / m

    @staticmethod
    def get_grains(labels):
        """ Segments classes labels into grain labels """
        new_labels = measure.label(labels, connectivity=2, background=0)
        new_labels = slu.relabel(new_labels)
        return new_labels
