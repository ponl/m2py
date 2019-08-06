import os

import numpy as np
from scipy import signal
from matplotlib import pyplot, colors
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from ssts.utils import config

INFO = config.data_info

NUM_COLS = 2  # number of cols in plots

ALPHA = 0.8  # transparency of labels in graphs
NUM_BINS = 30  # number of bins in histograms


## Property distribution and correlation methods


def show_property_distributions(data, data_type, outliers=None):
    """
    Plots the pdfs of the data properties before classification

    Parameters
    ----------
        data : NumPy Array
            SPM data supplied by the user
        data_type : str
            data type corresponding to config.data_info keyword (QNM, AMFM, cAFM)
        outliers : NumPy Array
            boolean, 2D array of outlier flags (1's) for functions to pass over
    
    Returns
    ----------
    
    """
    props = INFO[data_type]["properties"]

    h, w, c = data.shape
    num_plots = c
    num_cols = NUM_COLS
    num_rows = int(np.ceil(num_plots / num_cols))

    fig = pyplot.figure(figsize=(10, 15), dpi=80, facecolor="w", edgecolor="k")
    for j in range(c):
        if outliers is not None:
            x = [data[n, m, j] for n in range(h) for m in range(w) if not outliers[n, m]]
        else:
            x = [data[n, m, j] for n in range(h) for m in range(w)]

        pyplot.subplot(num_rows, num_cols, j + 1)
        pyplot.hist(x, NUM_BINS, alpha=ALPHA, density=True)
        pyplot.grid()
        pyplot.title(props[j])

    pyplot.tight_layout()
    pyplot.show()


def get_correlations(path):
    """
    Computes correlation for all files in path
    
    Parameters
    ----------
        path : str
            data directory
    
    Returns
    ----------
        cors :list
            correlation between all pairs of properties per file
    """
    files = os.listdir(path)
    N = len(files)
    cors = []
    for i in range(N):
        f = files[i]
        data = np.load(os.path.join(path, f))
        c = data.shape[2]

        C = np.zeros((c, c))  # correlation matrix of properties per file
        for j in range(c):
            for k in range(j, c):
                C[j, k] = C[k, j] = np.corrcoef(data[:, :, j].flatten(), data[:, :, k].flatten())[0][1]

        cors.append(C)

    return cors


def get_correlation_values(cors, r, c):
    """
    Gets correlation between properties r and c for all files
    
    Parameters
    ----------
        cors : list
            propertiess correlations per file
        r : int
            first property index
        c : int
            second property index
            
    Returns
    ----------
        rc_cors (list): correlation between properties r and c per file
    """
    rc_cors = [cor[r, c] for cor in cors]
    return rc_cors


def show_correlations(num_props, data_type, path):
    """
    Plots correlations between all properties
    
    Parameters
    ----------
        num_props : int
            number of properties
        data_type : str
            data type corresponding to config.data_info keyword (QNM, AMFM, cAFM)
        path : str
            data directory
        
    Returns
    ----------
    
    """
    props = INFO[data_type]["properties"]

    cors = get_correlations(path)

    fig = pyplot.figure(figsize=(20, 30))
    fig.subplots_adjust(hspace=1)

    cnt = 1
    for i in range(num_props):
        for j in range(num_props):
            pyplot.subplot(num_props, num_props, cnt)
            pyplot.title(f"{props[i]} -- {props[j]}", fontsize=11)
            if i == j:  # skip auto-correlation
                pyplot.xlabel("Correlation")
                pyplot.xlim(-1, 1)
                pyplot.grid()
                cnt += 1
                continue

            P = get_correlation_values(cors, i, j)
            V = [p for p in P if not np.isnan(p)]

            # TODO how many bins?
            pyplot.hist(V, bins=5, alpha=ALPHA, density=True)
            pyplot.xlabel("Correlation")
            pyplot.xlim(-1, 1)
            pyplot.grid()

            cnt += 1

    pyplot.show()


## Outlier detection and filtering methods


def extract_outliers(data, data_type, threshold=2.5):
    """
    Finds outliers from data

    Parameters
    ----------
        data : NumPy Array
            SPM data supplied by the user
        data_type : str
            data type corresponding to config.data_info keyword (QNM, AMFM, cAFM)
        threshold : float
            z-score threshold at which to flag a pixel as an outlier
        
    Returns
    ----------
        outliers : NumPy Array
            boolean, 2D array of outlier flags (1's) for functions to pass over
    """
    props = INFO[data_type]["properties"]
    if "Height" in props:
        height_index = props.index("Height")
    else:
        return None

    x = data[:, :, height_index]

    # Smooth data
    flt = np.array([[0.5, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 0.5]])
    y = signal.convolve2d(x, flt, boundary="symm", mode="same")

    # Compute z-scores
    u = np.mean(y)
    s = np.std(y)
    z = np.abs((u - y) / s)

    # Threshold by z-score
    return z > threshold


def show_outliers(data, data_type, outliers):
    """
    Plots data properties and outliers
    
    Parameters
    ----------    
        data : NumPy Array
            SPM data supplied by the user
        data_type : str
            data type corresponding to config.data_info keyword (QNM, AMFM, cAFM)
        outliers : NumPy Array
            boolean, 2D array of outlier flags (1's) for functions to pass over
        
    Returns
    ----------
    
    """
    props = INFO[data_type]["properties"]
    if "Height" in props:
        height_index = props.index("Height")
    else:
        return

    fig = pyplot.figure(figsize=(15, 4))

    pyplot.subplot(1, 3, 1)
    m = pyplot.imshow(data[:, :, height_index], aspect="auto")
    pyplot.title("Height")
    pyplot.colorbar(m, fraction=0.046, pad=0.04)

    pyplot.subplot(1, 3, 2)
    pyplot.imshow(outliers, aspect="auto")
    pyplot.title("Height Outliers")

    no_outliers_data = np.copy(data)
    height_data = no_outliers_data[:, :, height_index]
    height_data[outliers == 1] = np.mean(height_data)
    pyplot.subplot(1, 3, 3)
    m = pyplot.imshow(height_data, aspect="auto")
    pyplot.title("Height")
    pyplot.colorbar(m, fraction=0.046, pad=0.04)

    pyplot.tight_layout()
    pyplot.show()


def smooth_outliers_from_data(data, outliers):
    """
    Replaces outliers from each channel of data with their mean.
    
    Parameters
    ----------
        data : NumPy Array
            SPM data supplied by the user
        outliers : NumPy Array
            boolean, 2D array of outlier flags (1's) for functions to pass over
            
    Returns
    ----------
        no_outliers_data : NumPy Array
            SPM data with outlier values replaced with the channel's mean value
    """
    h, w, c = data.shape

    no_outliers_data = np.copy(data)
    for i in range(c):
        no_outliers_data[:, :, i][outliers == 1] = np.mean(no_outliers_data[:, :, i])

    return no_outliers_data


## Frequency removal
def apply_frequency_removal(data, data_type, compression_percent=95):
    """
    Removes small-magnitude frequencies from data
    
    Parameters
    ----------
        data : NumPy Array
            SPM data supplied by the user
        data_type : str
            data type corresponding to config.data_info keyword (QNM, AMFM, cAFM)
        compression_percent : float
            percentage of compression
            
    Returns
    ----------
        new_data : NumPy Array
            compressed data
    """

    def remove_small_magnitude_freqs(f_prop_shift, h, w, compression_percent):
        """
        Removes small-magnitude frequencies in Fourier space
        
        Parameters
        ----------
            f_prop_shift : NumPy Array
                frequencies
            h : int
                height of data array
            w : int
                width of data array
            compression_percent : float
                percentage of compression
            
        Returns
        ----------
            f_prop_shift : NumPy Array
                high frequencies
        """
        mags = np.abs(f_prop_shift)
        thresh = np.percentile(mags, compression_percent)
        cond = np.abs(f_prop_shift) < thresh
        f_prop_shift[cond] = 0
        return f_prop_shift

    props = INFO[data_type]["properties"]

    new_data = np.copy(data)

    h, w, c = data.shape
    num_plots = 3 * c
    num_cols = 3 * NUM_COLS
    num_rows = int(np.ceil(num_plots / num_cols))

    fig = pyplot.figure(figsize=(14, 21), dpi=80, facecolor="w", edgecolor="k")
    count = 1
    for k in range(c):
        prop = data[:, :, k]

        pyplot.subplot(num_rows, num_cols, count)
        pyplot.title(props[k])
        pyplot.imshow(prop)
        count += 1

        f_prop = fft2(prop)
        f_prop_shift = fftshift(f_prop)
        f_prop_shift = remove_small_magnitude_freqs(f_prop_shift, h, w, compression_percent)

        pyplot.subplot(num_rows, num_cols, count)
        pyplot.imshow(np.abs(f_prop_shift), norm=colors.LogNorm(vmin=np.mean(np.abs(f_prop_shift))))
        pyplot.title("Large-Magnitude Frequencies")
        count += 1

        f_prop = ifftshift(f_prop_shift)
        high_prop = np.real(ifft2(f_prop))

        pyplot.subplot(num_rows, num_cols, count)
        pyplot.title(props[k])
        pyplot.imshow(high_prop)
        count += 1

        new_data[:, :, k] = high_prop

    pyplot.tight_layout()
    pyplot.show()

    return new_data
