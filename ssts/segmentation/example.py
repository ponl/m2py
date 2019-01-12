import numpy as np
import segmentation_gmm as seg_gmm
from methods import main_methods as mm # NOTE SSTS directory must be in PYTHONPATh

which_example = 2

## NOTE Segmentation example without using outliers
if which_example == 0:

    # Location of data
    data_dir = "/Users/diegotorrejon/Projects/Other/SSTS/data/2componentfilms/combined/PTB7PC71BM_CBonly_ascast_fresh_500 nm_combined.npy"
    data = np.load(data_dir)

    # Initialize GMM segmenter
    seg = seg_gmm.SegmenterGMM(n_components=3)

    # Run segmentation
    labels = seg.fit_transform(data)

    # Plot classification
    mm.show_classification(labels, data)

    # Plot classification distributions
    mm.show_classification_distributions(labels, data)

## NOTE Segmentation example using outliers
if which_example == 1:

    data_dir = "/Users/diegotorrejon/Projects/Other/SSTS/data/nanowires/combined/100-0_48_NW-1_combined.npy"
    data = np.load(data_dir)

    # Initialize GMM segmented
    seg = seg_gmm.SegmenterGMM(n_componets=3)

    # Get outliers
    outliers = mm.extract_outliers(data)

    # Run segmentation
    labels = seg.fit_transform(data, outliers)

    # Plot classification
    mm.show_classification(labels, data)

    # Plot classification distributions
    mm.show_classification_distributions(labels, data)

## NOTE Segmentation example with dimensionality reduction (PCA) across physical properties
if which_example == 2:

    data_dir = "/Users/diegotorrejon/Projects/Other/SSTS/data/2componentfilms/combined/PTB7PC71BM_CBonly_ascast_fresh_500 nm_combined.npy"
    data = np.load(data_dir)

    # Initialize GMM segmented
    seg = seg_gmm.SegmenterGMM(n_components=2, embedding_dim=3)

    # Run segmentation
    labels = seg.fit_transform(data)

    # Plot classification
    mm.show_classification(labels, data)

    # Plot classification distributions
    mm.show_classification_distributions(labels, data)

## NOTE Segmentation example with dimensionality reduction (PCA) across neighboring pixels and physical properties
if which_example == 3:

    data_dir = "/Users/diegotorrejon/Projects/Other/SSTS/data/2componentfilms/combined/PTB7PC71BM_CBonly_ascast_fresh_500 nm_combined.npy"
    data = np.load(data_dir)

    # Initialize GMM segmented
    seg = seg_gmm.SegmenterGMM(n_components=3, padding=3, embedding_dim=10, zscale=True)

    # Run segmentation
    labels = seg.fit_transform(data)

    # Plot classification
    mm.show_classification(labels, data)

    # Plot classification distributions
    mm.show_classification_distributions(labels, data)

