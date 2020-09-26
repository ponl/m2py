import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import pyplot, colors, cm
from skimage import morphology, measure

from m2py.utils import config
from m2py.utils import pre_processing
from m2py.utils import seg_label_utils as slu

INFO = config.data_info

ALPHA = 0.8  # transparency of labels in graphs
NUM_BINS = 30  # number of bins in histograms

NUM_COLS = 2  # number of cols in plots


def store_results(labels, output_file):
    """
    Store results.

    Args:
        labels (NumPy Array): matrix of classification per pixel
        output_file (str): output file for results
    """
    np.save(output_file, labels)


## Plotting methods


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


def get_metrics_df(grain_labels, gmm_labels, metrics):
    """
    Returns a dataframe of metrics for the grains

    Parameters
    ----------
        grain_labels : NumPy Array
            matrix of grain extraction per pixel
        gmm_labels : NumPy Array
            matrix of phase classification per pixel
        metrics : list
            list of metrics to extract in df

    Returns
    ----------

    """
    # NOTE: this ignores 0 which is ideal since it denotes background
    labels_metrics = measure.regionprops_table(grain_labels, properties=metrics)

    # Add phase information to df
    mean_labels = []
    ls = labels_metrics["label"]
    for l in ls:
        mean_label = stats.mode(gmm_labels[grain_labels == l])[0][0]
        mean_labels.append(mean_label)

    mean_labels = np.array(mean_labels)
    labels_metrics["phase"] = mean_labels

    labels_metrics_df = pd.DataFrame.from_dict(labels_metrics, orient="columns")
    return labels_metrics_df


def show_grain_metrics_distributions(df, in_meters=False, data_type=None):
    """
    Shows a histogram of each metric provided for the grains

    Parameters
    ----------
        df : Pandas Dataframe
            matrix of grain extraction per pixel
        in_meters : str
            specifies whether to use physical units or not
        data_type : str
            data type as specified by config file

    Returns
    ----------

    """
    for i, l in enumerate(df.phase.unique()):  # per phase
        fig = pyplot.figure(figsize=(12, 8), dpi=80, facecolor="w", edgecolor="k")

        counter = 1
        for c in df.columns:
            if c in ["label", "phase"]:
                continue

            pyplot.subplot(2, 3, counter)
            filtered_df = df[df["phase"] == l]

            if in_meters and data_type:  # in physical units
                pixel_size = INFO[data_type]["pixel_size"]
                sample_area = INFO[data_type]["sample_area"]
                if c in ["perimeter", "minor_axis_length", "major_axis_length"]:
                    data = filtered_df[c] * pixel_size
                elif c in ["area"]:
                    percentages = filtered_df[c] / filtered_df[c].sum()
                    data = percentages * sample_area
            else:  # in pixel units
                data = filtered_df[c]

            pyplot.hist(data, bins=10, density=True)

            pyplot.xlabel(" ".join(c.split("_")).capitalize())
            pyplot.grid()
            counter += 1

        pyplot.suptitle(f"Metrics for Phase: {l}", y=1, color="red")
        pyplot.tight_layout()
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


def get_gmm_background_label(gmm_labels, connected_labels):
    """
    Gets the GMM backgound comoponent using connected components information. The background component will
    be the one with the fewest connected components.

    Parameters
    ---------
        gmm_labels : NumPy Array
            GMM labeling
        connected_labels : NumPy Array
            connected components labeling

    Returns
    ----------
        bg_label : int
            GMM background label

    """

    return min(list(np.unique(gmm_labels)), key=lambda k: np.size(np.unique(connected_labels[gmm_labels == k])))


def applies_morphological_cleanup(labels, bg_label, morph_radius=5, morph_min_size=25):
    """
    Applies morphological cleanup to labeling assignment per component / grain.

    Parameters
    ---------
        labels : NumPy Array
            labeling assignment
        bg_label : int
            GMM background label
        morph_radius : int
            disk radius for the morphological orperation
        morph_min_size : int
            remove small objects smaller than this value

    Returns
    ----------
        clean_labels : NumPy Array
            labeling after morphological cleanup

    """
    components = slu.get_unique_labels(labels)
    components.remove(bg_label)

    # Initialize clean labels as bacdground
    clean_labels = (labels == bg_label) * bg_label

    for i in components:  # non-background gmm components in increasing order
        gmm_labels = labels == i

        clean_gmm_labels = morphology.binary_opening(gmm_labels, morphology.disk(morph_radius))
        clean_gmm_labels = morphology.remove_small_objects(clean_gmm_labels, min_size=morph_min_size)
        clean_gmm_labels = clean_gmm_labels.astype(np.uint8)

        clean_labels[clean_gmm_labels == 1] = i
        clean_labels[(clean_gmm_labels == 0) * gmm_labels] = bg_label

    return clean_labels


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
