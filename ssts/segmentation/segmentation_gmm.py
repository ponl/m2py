import logging
from collections import Counter
from sklearn.mixture import GaussianMixture
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

        if len(outliers.shape) != 2:
            logger.warning("Outlier arrays must be of shape (height, width).")
            return None


        h, w, c = data.shape
        n = h * w

        data = data.reshape(c, n)

        if outliers is not None:
            outliers = outliers.reshape(c, n)

        data = SegmenterGMM.remove_outliers(data, outliers)

        if self.normalize:
            data = SegmenterGMM.normalize_data(data)

        if self.padding > 0:
            data = SegmenterGMM.get_windows(data, padding=self.padding)
            # Extract neighborhood
            # Flatten

        if self.embedding_dim is not None:
            self.pca = PCA(n_components=embedding_dim)
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
            labels (NumPy Array): The segmented array. Each pixel receives
                a label corresponding to its segment, of shape (height, width).
        """
        if self.gmm is None:
            logger.warning("Attempting to transform prior to fitting. You must call .fit() first.")
            return None

        pass


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
    def normalize_data(data):
        means = np.mean(data, axis=1)
        stds = np.std(data, axis=1)
        return (data - means) / stds


    @staticmethod
    def remove_outliers(data, outliers):
        if outliers is None:
            return data

        n_outlier = len(outliers)
        return np.array([data[i] if not outliers[i] for i in range(n_outliers)])


    @staticmethod
    def get_windows(data, padding=3):

