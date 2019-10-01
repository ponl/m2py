import logging

from matmap.segmentation.segmentation_gmm import SegmenterGMM


logger = logging.getLogger(__name__)

def segment(data, outliers=None, method="gmm", **kwargs):
    """
    Interface for segmenting material property arrays. Supported methods
    are the following:
        gmm: Gaussian Mixture Model Segmentation (SegmenterGMM)

    Parameters
    ----------
        data : NumPy Array
            The material properties array of shape (height, width, n_properties)
        outliers : NumPy Array
            The boolean outlier matrix of shape (height, width)
        method : string
            One of the above supported methods.

    Returns
    ----------
        labels : NumPy Array
            Label array describing the segments, of shape (height, width)
    """
    if method == "gmm":
        return SegmenterGMM(**kwargs).fit_transform(data, outliers)
    else:
        logger.warning("The specified method ({0}) is not supported".format(method))
        return None

