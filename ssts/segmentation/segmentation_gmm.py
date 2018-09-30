import logging
import numpy as np
from collections import Counter
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class SegmenterGMM(object):
    def __init__(self, n_components=3, normalize=True, embedding_dim=None, padding=0):
        """

        Args:
            n_components (int): Number of components (i.e. segments) to learn in the GMM.
            normalize (bool): Normalize the material properties before processing.
        """

        self.normalize = normalize
        self.embedding_dim = embedding_dim
        self.n_components = n_components
        self.padding = padding

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

        if outliers and len(outliers.shape) != 2:
            logger.warning("Outlier arrays must be of shape (height, width).")
            return None

        h, w, c = data.shape
        n = h * w

        if self.padding > 0:
            print("Building Windows")
            data = SegmenterGMM.get_windows(data, padding=self.padding)
        else:
            data = data.reshape(n, c)

        if outliers is not None:
            print("Removing Outliers")
            outliers = outliers.reshape(n, c)

        data = SegmenterGMM.remove_outliers(data, outliers)

        if self.normalize:
            print("Scaling data")
            self.sst = StandardScaler()
            data = self.sst.fit_transform(data)

        if self.embedding_dim is not None:
            print("Fitting PCA")
            self.pca = PCA(n_components=self.embedding_dim)
            data = self.pca.fit_transform(data)

        print("Fitting GMM")
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
            labels (NumPy Array): The segmented array. Each pixel receives
                a label corresponding to its segment, of shape (height, width).
        """
        if self.gmm is None:
            logger.warning("Attempting to transform prior to fitting. You must call .fit() first.")
            return None

        h, w, c = data.shape
        n = h * w

        if self.padding > 0:
            print("Building Windows")
            data = SegmenterGMM.get_windows(data, padding=self.padding)
        else:
            data = data.reshape(n, c)

        if outliers is not None:
            print("Removing Outliers")
            outliers = outliers.reshape(n, c)

        data = SegmenterGMM.remove_outliers(data, outliers)

        if self.normalize:
            print("Scaling Data")
            data = self.sst.transform(data)

        if self.embedding_dim is not None:
            print("PCA Transform")
            data = self.pca.transform(data)

        print("Predicting with GMM")
        labels = self.gmm.predict(data)

        return np.reshape(labels, (h, w))

    def fit_transform(self, data, outliers=None):
        """
        Learns and applies a learned Gaussian mixture model and
        returns a labelled array.

        Args:
            data (NumPy Array): Material properties array of shape (height, width, n_properties)
            outliers (NumPy Array): Binary array indicating outliers of shape (height, width)

        Returns:
            labels (NumPy Array): The segmented array. Each pixel receives
                a label corresponding to its segment, of shape (height, width).
        """
        gmm = self.fit(data, outliers)

        if gmm is None:
            return None

        return self.transform(data, outliers)


    @staticmethod
    def remove_outliers(data, outliers):
        if outliers is None:
            return data

        n_outlier = len(outliers)
        return np.array([data[i] for i in range(n_outliers) if not outliers[i]])


    @staticmethod
    def get_windows(data, padding=3, flatten=True):
        h, w, c = data.shape
        n = 2 * padding + 1
        wins = np.zeros((h * w, n**2, c))
        for d in range(data.shape[2]):
            X = np.pad(data[:,:,d], padding, "constant")
            idx = 0
            for i in range(padding, data.shape[0] + padding):
                for j in range(padding, data.shape[1] + padding):
                    imin, imax = i - padding, i + padding + 1
                    jmin, jmax = j - padding, j + padding + 1
                    wins[idx, :, d] = X[imin:imax, jmin:jmax].flatten()
                    idx += 1

        if flatten:
            l = wins.shape[1]
            wins = np.reshape(wins, (-1, l * c))

        return wins

