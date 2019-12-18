import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, colors, cm

from m2py.utils import config
from m2py.utils import pre_processing
from m2py.utils import seg_label_utils as slu

INFO = config.data_info

ALPHA = 0.8  # transparency of labels in graphs
NUM_BINS = 30  # number of bins in histograms

NUM_COLS = 2  # number of cols in plots


## Plotting methods


def store_results(labels, output_file):
    """
    Store results.

    Args:
        labels (NumPy Array): matrix of classification per pixel
        output_file (str): output file for results
    """
    np.save(output_file, labels)


def show_classification(labels, data, data_type, input_cmap="jet"):
    """
    Shows classification of pixels after segmentation

    Parameters
    ----------
        labels : NumPy Array
            matrix of classification per pixel
        data : NumPy Array
            matrix containing user entered SPM data
        data_type : str
            string designating data type (QNM, AMFM, cAFM)
        input_cmap : str
            string designating matplotlib colormap to use

    Returns
    ----------
    """
    props = INFO[data_type]["properties"]

    unique_labels = slu.get_unique_labels(labels)
    num_labels = len(unique_labels)

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
        ax_r.imshow(labels, alpha=ALPHA, cmap=cmap)

        colorbar_index(ncolors=num_labels, cmap=cmap)

    pyplot.tight_layout()
    pyplot.show()


def show_classification_distributions(labels, data, outliers, data_type, title_flag=True):
    """
    Shows distributions of classes after segmentation

    Parameters
    ----------
        labels : NumPy Array
            matrix of classification per pixel
        data : NumPy Array
            SPM data supplied by the user
        outliers : NumPy Array
            array of outlier pixels
        data_type : str
            data type corresponding to config.data_info keyword (QNM, AMFM, cAFM)
        title_flag : bool
            flag for plots to show titles or not

    Returns
    ----------
    """
    props = INFO[data_type]["properties"]

    if outliers is None:
        outliers = 0

    unique_labels = slu.get_unique_labels(labels)
    num_labels = len(unique_labels)

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
        for index, j in enumerate(unique_labels):
            color_step = num_labels - (index + 1)
            labels_wo_outliers = labels * (1 - outliers)
            class_data = data[:, :, i][labels_wo_outliers == j]
            ax_r.hist(class_data, NUM_BINS, alpha=ALPHA, density=True, color=cmap(color_step))

    pyplot.tight_layout()
    pyplot.show()


def show_grain_area_distribution(labels, data_type, data_subtype):
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
    sample_area = INFO[data_type]["sample_size"][data_subtype] ** 2

    unique_labels = slu.get_unique_labels(labels)
    grain_areas = sorted([np.sum(labels == l) for l in unique_labels], reverse=True)

    percent_grain_areas = grain_areas / np.sum(grain_areas) * 100
    physical_grain_areas = percent_grain_areas / 100 * sample_area

    pyplot.figure(figsize=(18, 5), dpi=80, facecolor="w", edgecolor="k")
    pyplot.subplot(1, 3, 1)
    pyplot.plot(np.log10(grain_areas), "ro")
    pyplot.plot(np.log10(grain_areas), "b")
    pyplot.xlabel("Grain")
    pyplot.ylabel("Log number of pixles per grain")
    pyplot.title("Log Number of Pixels per Grain")
    pyplot.grid()

    pyplot.subplot(1, 3, 2)
    pyplot.plot(percent_grain_areas, "ro")
    pyplot.plot(percent_grain_areas, "b")
    pyplot.xlabel("Grain")
    pyplot.ylabel("Grain percentage (%)")
    pyplot.title("Grain Percentage (%)")
    pyplot.grid()

    pyplot.subplot(1, 3, 3)
    pyplot.plot(physical_grain_areas, "ro")
    pyplot.plot(physical_grain_areas, "b")
    pyplot.xlabel("Grain")
    pyplot.ylabel("Grain area (um2)")
    pyplot.title("Grain Area (um2)")
    pyplot.grid()

    pyplot.show()


def show_distributions_together(labels, data, data_type, input_cmap):
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
        input_cmap : str
            specifies which color map to use
            
    Returns
    ----------
    
    """
    props = INFO[data_type]["properties"]

    unique_labels = slu.get_unique_labels(labels)
    num_labels = len(unique_labels)

    h, w, c = data.shape
    num_plots = 2 * c
    num_cols = 2 * NUM_COLS
    num_rows = int(np.ceil(num_plots / num_cols))

    fig = pyplot.figure(figsize=(20, 15), dpi=80, facecolor="w", edgecolor="k")
    cnt = 1
    cmap = pyplot.get_cmap(input_cmap, num_labels)
    for i in range(c):
        ax_l = pyplot.subplot(num_rows, num_cols, cnt)
        cnt += 1
        ax_l.set_title("Segmentation")
        ax_l.imshow(labels, alpha=ALPHA, cmap=cmap, aspect="auto")

        ax_r = pyplot.subplot(num_rows, num_cols, cnt)
        cnt += 1
        ax_r.set_title(props[i])
        ax_r.grid()

        for index, j in enumerate(unique_labels):  # plots mask and distribution per class
            color_step = num_labels - (index + 1)
            ax_r.hist(data[:, :, i][labels == j], NUM_BINS, alpha=ALPHA, density=True, color=cmap(color_step))

    pyplot.tight_layout()
    pyplot.show()


def show_overlaid_distribution(probs, data, data_type, outliers=None):
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
        outliers : NumPy Array
            array of outlier pixels

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
        prob = probs[:, :, i]
        masked = np.ma.masked_where(outliers == 1, prob)
        m = ax.imshow(masked, vmin=0, vmax=1)
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
        title_flag : bool
            flag for plots to show titles or not
        sample_flag : bool
            flag to sample points for visualization

    Returns
    ----------
    
    """
    props = INFO[data_type]["properties"]

    unique_labels = slu.get_unique_labels(labels)
    num_labels = len(unique_labels)

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
            for index, l in enumerate(unique_labels):
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
    ----------
        cmap : str
            colormap instance.
        N : int
            number of colors.
            
    Returns
    ----------
        discretized version of continuous colormap
    """
    colors_i = np.concatenate((np.linspace(0, 1.0, N), (0.0, 0.0, 0.0, 0.0)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1.0, N + 1)
    cdict = {}
    for ki, key in enumerate(("red", "green", "blue")):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in range(N + 1)]

    return colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict)
