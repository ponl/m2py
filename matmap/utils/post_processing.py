import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, colors, cm

from matmap.utils import config
from matmap.utils import pre_processing
from matmap.utils import seg_label_utils as slu

INFO = config.data_info

LABEL_THRESH = 4 # each label must have more than this number of pixels
BG_THRESH = 10000

ALPHA = 0.8  # transparency of labels in graphs
NUM_BINS = 30  # number of bins in histograms

NUM_COLS = 2  # number of cols in plots

data_channels = config.data_info['QNM']['properties'] # channel names to use


## Plotting methods

def show_classification(labels, data, data_type, input_cmap='jet', bg_contrast_flag=False):
    """ Shows classification of pixels after segmentation
    Args:
        labels (np.array): matrix of classification per pixel
        data (np.array): data
        data_type (srt): data type (QNM, AMFM, cAFM)
        input_cmap (str): to use different color map
        bg_contrast_flag (bool) highlights biggest grain (background) in plot
    """
    props = INFO[data_type]["properties"]

    unique_labels = slu.get_unique_labels(labels)
    grain_labels = [l for l in unique_labels if np.sum(labels == l) > LABEL_THRESH]
    bg_labels = [l for l in grain_labels if np.sum(labels == l) > BG_THRESH]
    num_labels = len(grain_labels)

    h, w, c = data.shape
    num_plots = 2 * c
    num_cols = 2 * NUM_COLS
    num_rows = int(np.ceil(num_plots / num_cols))

    fig = pyplot.figure(figsize=(20, 14), dpi=80, facecolor="w", edgecolor="k")
    cnt = 1
    cmap = pyplot.get_cmap(input_cmap, num_labels)
    for i in range(c):
        ax_l = pyplot.subplot(num_rows, num_cols, cnt)
        cnt += 1
        ax_l.set_title(props[i])
        m = ax_l.imshow(data[:, :, i])
        pyplot.colorbar(m, fraction=0.046, pad=0.04)

        ax_r = pyplot.subplot(num_rows, num_cols, cnt)
        cnt += 1
        ax_r.set_title("Segmentation")
        for index, j in enumerate(grain_labels):  # plots mask per class
            if (j in bg_labels) and bg_contrast_flag:
                color_step = num_labels - 1 # uses a distinct color
            else:
                color_step = num_labels - (index + 1)

            # mask hides the values of the condition
            mask = np.ma.masked_where(labels != j, color_step * np.ones(labels.shape))
            ax_r.imshow(mask, alpha=ALPHA, cmap=cmap, vmin=0, vmax=cmap.N)

        colorbar_index(ncolors=num_labels, cmap=cmap)

    pyplot.tight_layout()
    pyplot.show()


def show_classification_distributions(labels, data, data_type, title_flag=True):
    """
    Shows distributions of classes after segmentation
    
    Parameters
    ----------
        labels : NumPy Array
            matrix of classification per pixel
        data : NumPy Array
            SPM data supplied by the user
        data_type : str
            data type corresponding to config.data_info keyword (QNM, AMFM, cAFM)
        title_flag : bool
            flag for plots to show titles or not
            
    Returns
    ----------
    
    """
    props = INFO[data_type]["properties"]

    unique_labels = slu.get_unique_labels(labels)
    grain_labels = [l for l in unique_labels if np.sum(labels == l) > LABEL_THRESH]
    num_labels = len(grain_labels)

    h, w, c = data.shape
    num_plots = 2 * c
    num_cols = 2 * NUM_COLS
    num_rows = int(np.ceil(num_plots / num_cols))

    fig = pyplot.figure(figsize=(20, 15), dpi=80, facecolor="w", edgecolor="k")
    cnt = 1
    cmap = pyplot.get_cmap("jet", num_labels)
    for index_i, i in enumerate(range(c)):
        ax_l = pyplot.subplot(num_rows, num_cols, cnt)
        cnt += 1
        if title_flag:
            ax_l.set_title(props[i])
        else:
            ax_l.set_title(f"PCA component {index_i + 1}")

        m = ax_l.imshow(data[:, :, i], aspect="auto")
        pyplot.colorbar(m, fraction=0.046, pad=0.04)

        ax_r = pyplot.subplot(num_rows, num_cols, cnt)
        cnt += 1
        ax_r.grid()
        ax_r.set_title("Distributions")
        for index, j in enumerate(grain_labels):
            color_step = num_labels - (index + 1)
            ax_r.hist(data[:, :, i][labels == j], NUM_BINS, alpha=ALPHA, density=True, color=cmap(color_step))

    pyplot.tight_layout()
    pyplot.show()


def show_grain_area_distribution(labels, data_type, data_subtype=None):
    """
    Computes a histogram of the number of pixels per label
    
    Parameters
    ----------
        labels : NumPy Array
            matrix of classification per pixel
        data : NumPy Array
            SPM data supplied by the user
        data_type : str
            data type corresponding to config.data_info keyword (QNM, AMFM, cAFM)
            
    Returns
    ----------
    
    """
    if data_subtype is None:
        sample_area = INFO[data_type]["sample_size"] ** 2
    else:
        sample_area = INFO[data_type]["sample_size"][data_subtype] ** 2

    unique_labels = slu.get_unique_labels(labels)
    grain_areas = [np.sum(labels == l) for l in unique_labels]

    sig_areas = sorted([a for a in grain_areas if a > LABEL_THRESH], reverse=True)
    percent_sig_areas = sig_areas / np.sum(grain_areas) * 100
    physical_sig_areas = percent_sig_areas / 100 * sample_area

    pyplot.figure(figsize=(18, 5), dpi=80, facecolor="w", edgecolor="k")
    pyplot.subplot(1, 3, 1)
    pyplot.plot(np.log10(sig_areas), "ro")
    pyplot.plot(np.log10(sig_areas), "b")
    pyplot.xlabel("Grain")
    pyplot.ylabel("Log number of pixles per grain")
    pyplot.title("Log Number of Pixels per Grain")
    pyplot.grid()

    pyplot.subplot(1, 3, 2)
    pyplot.plot(percent_sig_areas, "ro")
    pyplot.plot(percent_sig_areas, "b")
    pyplot.xlabel("Grain")
    pyplot.ylabel("Grain percentage (%)")
    pyplot.title("Grain Percentage (%)")
    pyplot.grid()

    pyplot.subplot(1, 3, 3)
    pyplot.plot(physical_sig_areas, "ro")
    pyplot.plot(physical_sig_areas, "b")
    pyplot.xlabel("Grain")
    pyplot.ylabel("Grain area (um2)")
    pyplot.title("Grain Area (um2)")
    pyplot.grid()

    pyplot.show()


def show_distributions_together(labels, data, data_type):
    """
    Shows distributions of classes after segmentation
    
    Parameters
    ----------
        labels : NumPy Array
            matrix of classification per pixel
        data : NumPy Array
            SPM data supplied by the user
        data_type : str
            data type corresponding to config.data_info keyword (QNM, AMFM, cAFM)
            
    Returns
    ----------
    
    """
    props = INFO[data_type]["properties"]

    unique_labels = slu.get_unique_labels(labels)
    grain_labels = [l for l in unique_labels if np.sum(labels == l) > LABEL_THRESH]
    num_labels = len(grain_labels)

    h, w, c = data.shape
    num_plots = 2 * c
    num_cols = 2 * NUM_COLS
    num_rows = int(np.ceil(num_plots / num_cols))

    fig = pyplot.figure(figsize=(20, 15), dpi=80, facecolor="w", edgecolor="k")
    cnt = 1
    cmap = pyplot.get_cmap("jet", num_labels)
    for i in range(c):
        ax_l = pyplot.subplot(num_rows, num_cols, cnt)
        cnt += 1
        ax_l.set_title("Segmentation")

        ax_r = pyplot.subplot(num_rows, num_cols, cnt)
        cnt += 1
        ax_r.set_title(props[i])
        ax_r.grid()

        for index, j in enumerate(grain_labels):  # plots mask and distribution per class
            color_step = num_labels - (index + 1)
            mask = np.ma.masked_where(labels != j, color_step * np.ones(labels.shape))
            ax_l.imshow(mask, alpha=ALPHA, cmap=cmap, aspect="auto", vmin=0, vmax=num_labels)

            ax_r.hist(data[:, :, i][labels == j], NUM_BINS, alpha=ALPHA, density=True, color=cmap(color_step))

    pyplot.tight_layout()
    pyplot.show()


def show_distributions_separately(labels, data, data_type):
    """
    Shows distributions of each class separately
    
    Parameters
    ----------
        labels : NumPy Array
            matrix of classification per pixel
        data : NumPy Array
            SPM data supplied by the user
        data_type : str
            data type corresponding to config.data_info keyword (QNM, AMFM, cAFM)
            
    Returns
    ----------
    
    """
    props = INFO[data_type]["properties"]

    unique_labels = slu.get_unique_labels(labels)
    grain_labels = [l for l in unique_labels if np.sum(labels == l) > LABEL_THRESH]
    num_labels = len(grain_labels)

    h, w, c = data.shape
    num_plots = 2 * c
    num_cols = 2 * NUM_COLS
    num_rows = int(np.ceil(num_plots / num_cols))

    cmap = pyplot.get_cmap("jet", num_labels)
    for index, gl in enumerate(grain_labels):
        color_step = num_labels - (index + 1)
        fig = pyplot.figure(figsize=(20, 15), dpi=80, facecolor="w", edgecolor="k")
        cnt = 1
        for i in range(c):
            ax_l = pyplot.subplot(num_rows, num_cols, cnt)
            cnt += 1
            ax_l.set_title(props[i])
            m = ax_l.imshow(data[:, :, i], aspect="auto")
            mask = np.ma.masked_where(labels == gl, np.ones(labels.shape))
            ax_l.imshow(mask, alpha=1, cmap="bone", aspect="auto", vmin=0, vmax=1)
            pyplot.colorbar(m, fraction=0.046, pad=0.04)

            ax_r = pyplot.subplot(num_rows, num_cols, cnt)
            cnt += 1
            ax_r.set_title("Distribution")
            ax_r.hist(data[:, :, i][labels == gl], NUM_BINS, alpha=ALPHA, density=True, color=cmap(color_step))
            ax_r.grid()

        pyplot.tight_layout()
        pyplot.show()


def show_overlaid_distribution(probs, data, data_type):
    """
    Plots distributions overlaid on pixels
    Parameters
    ----------
        probs : NumPy Array
            array of probabilities per pixel
        data : NumPy Array
            SPM data supplied by the user
        data_type : str
            data type corresponding to config.data_info keyword (QNM, AMFM, cAFM)
            
    Returns
    ----------
    
    """
    props = INFO[data_type]["properties"]

    h, w, n = probs.shape
    h, w, c = data.shape

    num_plots = n + c
    num_cols = 2 * NUM_COLS
    num_rows = int(np.ceil(num_plots / num_cols))

    fig = pyplot.figure(figsize=(20, 15), dpi=80, facecolor="w", edgecolor="k")
    cnt = 1
    for i in range(c):
        ax = pyplot.subplot(num_rows, num_cols, cnt)
        cnt += 1
        ax.set_title(props[i])
        m = ax.imshow(data[:, :, i])
        pyplot.colorbar(m, fraction=0.046, pad=0.04)

    for i in range(n):
        ax = pyplot.subplot(num_rows, num_cols, cnt)
        cnt += 1
        m = ax.imshow(probs[:, :, i], vmin=0, vmax=1)
        pyplot.colorbar(m, fraction=0.046, pad=0.04)

    pyplot.show()


def show_classification_correlation(labels, data, data_type, title_flag=True, sample_flag=True):
    """
    Plots the correlation of data properties after classification
    
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

    unique_labels = slu.get_unique_labels(labels)
    grain_labels = [l for l in unique_labels if np.sum(labels == l) > LABEL_THRESH]
    num_labels = len(grain_labels)

    h, w, c = data.shape
    num_plots = np.sum(np.arange(c))
    num_cols = 2 * NUM_COLS
    num_rows = int(np.ceil(num_plots / num_cols))

    fig = pyplot.figure(figsize=(20, 15), dpi=80, facecolor="w", edgecolor="k")
    cmap = pyplot.get_cmap("jet", num_labels)
    cnt = 1
    for index_i, i in enumerate(range(c)):
        for index_j, j in enumerate(range(c)):
            if index_j <= index_i:
                continue

            ax = pyplot.subplot(num_rows, num_cols, cnt)
            ax.grid()
            if title_flag:
                ax.set_xlabel(props[index_i])
                ax.set_ylabel(props[index_j])
            else:
                ax.set_xlabel(f"PCA component {index_i + 1}")
                ax.set_ylabel(f"PCA component {index_j + 1}")

            cnt += 1
            title = f"Correlation: "
            for index, l in enumerate(grain_labels):
                data_i = data[:, :, index_i][labels == l]
                data_i /= np.max(np.abs(data_i))
                data_j = data[:, :, index_j][labels == l]
                data_j /= np.max(np.abs(data_j))
                corr = np.corrcoef(data_i, data_j)[0, 1]

                if sample_flag:
                    indices = np.arange(data_i.shape[0])
                    sample_size = min(1000, len(indices))
                    sampling = np.random.choice(indices, size=sample_size, replace=False)
                    data_i = np.take(data_i, sampling)
                    data_j = np.take(data_j, sampling)

                color_step = num_labels - (index + 1)
                ax.scatter(data_i, data_j, color=cmap(color_step), alpha=0.2)
                title += f"{index}: {np.round(corr, 3)} "

            ax.set_title(title)

    pyplot.tight_layout()
    pyplot.show()


# NOTE Auxilliary methods for creating discrete colorbar
def colorbar_index(ncolors, cmap):
    """
    Adds discrete colorbar to plot.
    
    Parameters
    ----------
        ncolors : int
            number of colors.
        cmap : str
            colormap instance
            
    Returns
    ----------
    
    """
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors + 0.5)
    colorbar = pyplot.colorbar(mappable, fraction=0.046, pad=0.04)
    if ncolors < 20:
        colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
        colorbar.set_ticklabels(range(ncolors))


def cmap_discretize(cmap, N):
    """
    Return a discrete colormap from the continuous colormap cmap.
    
    Parameters
        cmap : str
            colormap instance.
        N : int
            umber of colors.
    """
    colors_i = np.concatenate((np.linspace(0, 1.0, N), (0.0, 0.0, 0.0, 0.0)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1.0, N + 1)
    cdict = {}
    for ki, key in enumerate(("red", "green", "blue")):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]

    return colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict)


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
    N = pre_processing.get_sample_count(path)
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

