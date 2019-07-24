import numpy as np
from matplotlib import pyplot

from utils import config
from utils import io_and_data as iod

INFO = config.data_info

LABEL_THRESH = 5  # each label must have more than this number of pixels

ALPHA = 0.8  # transparency of labels in graphs
NUM_BINS = 30  # number of bins in histograms

NUM_COLS = 2  # number of cols in plots

## Properties Distributions
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

## Features Correlations



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
    N = iod.get_sample_count(path)
    cors = []
    for i in range(N):
        data = iod.get_data(i, path)
        c = data.shape[2]

        C = np.zeros((c, c))  # correlation matrix of properties per file
        for j in range(c):
            for k in range(j, c):
                C[j, k] = C[k, j] = np.corrcoef(data[:, :, j].flatten(),
                                                data[:, :, k].flatten())[0][1]

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


# TODO use with many data files
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


