import argparse

import numpy as np
import ssts.segmentation.segmentation_gmm as seg_gmm
import ssts.segmentation.segmentation_watershed as seg_water
from methods import main_methods as mm  # NOTE SSTS directory must be in PYTHONPATH

Z_SCORE_THRESH = 2.5

def main(data_path=None, example_number=None):
    if data_path is not None and example_number is not None:
        args = argparse.Namespace(data_path=data_path, example_number=example_number)
    else:
        p = argparse.ArgumentParser(description="""A script with examples on how to use segmentation code.""")
        p.add_argument("data_path", help="Path to data.")
        p.add_argument("example_number", help="Specifies which example to run.")
        args = p.parse_args()

    example_number = args.example_number
    data_file = args.data_path
    data_type = data_file.split("/")[-3]
    data_dir = "/".join(data_file.split("/")[:-1])
    data = np.load(data_file)

    ## NOTE Segmentation example without using outliers
    if example_number == "0":

        # Initialize GMM segmentation
        seg = seg_gmm.SegmenterGMM(n_components=2)

        # Run segmentation
        labels = seg.fit_transform(data)

        # Plot classification
        mm.show_classification(labels, data)

        # Plot classification distributions
        mm.show_classification_distributions(labels, data)

        # Overlay distributions on pixels
        probs = seg.get_probabilities(data)
        mm.show_overlaid_distribution(probs, data)

    ## NOTE Segmentation example using outliers
    elif example_number == "1":

        # Get outliers
        outliers = mm.extract_outliers(data, threshold=Z_SCORE_THRESH)
        mm.show_outliers(data, outliers)
        no_outliers_data = mm.smooth_outliers_from_data(data, outliers)

        # Shows a-priori property distributions
        mm.show_property_distributions(data, outliers)

        # Initialize GMM segmentation
        seg = seg_gmm.SegmenterGMM(n_components=2)

        # Run segmentation
        labels = seg.fit_transform(data, outliers)

        # Plot classification without outliers
        mm.show_classification(labels, no_outliers_data)  # uses data without outliers

        # Plot classification distribution
        mm.show_classification_distributions(labels, no_outliers_data)

        # Plot classification distributions with masks
        mm.show_distributions_together(labels, no_outliers_data)  # all classes plotted together
        mm.show_distributions_separately(labels, no_outliers_data)  # each class plotted per plot

    ## NOTE Segmentation example with dimensionality reduction (PCA) across physical properties AND grain segmentation
    elif example_number == "2":

        # Initialize GMM segmentation
        num_pca_components = 3
        seg = seg_gmm.SegmenterGMM(n_components=2, embedding_dim=num_pca_components)

        # Run segmentation
        pre_labels = seg.fit_transform(data)

        # Plot classification
        mm.show_classification(pre_labels, data)

        # Plot classification distributions
        mm.show_classification_distributions(pre_labels, data)

        # Handling PCA components from data
        h, w, c = data.shape
        n = h * w
        pca_components = seg.pca.transform(data.reshape(n, c))  # NOTE PCA was trained while fitting pre_seg
        pca_components = pca_components.reshape(h, w, num_pca_components)  # shape (512, 512, num_pca_components)

        # Plot classification distributions of PCA components
        mm.show_classification_distributions(pre_labels, pca_components, title_flag=False)

        # Show correlation after classification of PCA components
        mm.show_classification_correlation(pre_labels, pca_components, title_flag=False)

        # Create unique masks per grain
        post_labels = seg.get_grains(pre_labels)

        # Plot grain classification
        mm.show_classification(post_labels, data)

        # Plot area distribution of grains
        mm.show_grain_area_distribution(post_labels, data_type)

        # Plot distributions of segmented grains
        mm.show_distributions_together(post_labels, data)  # all grains plotted together
        mm.show_distributions_separately(post_labels, data)  # each grain plotted per plot

    ## NOTE Segmentation example with dimensionality reduction (PCA) across neighboring pixels and physical properties
    elif example_number == "3":

        # Get outliers
        outliers = mm.extract_outliers(data, threshold=Z_SCORE_THRESH)
        no_outliers_data = mm.smooth_outliers_from_data(data, outliers)

        # Initialize GMM segmentation # TODO need to optimize these parameters
        seg = seg_gmm.SegmenterGMM(n_components=3, padding=3, embedding_dim=10, zscale=True)

        # Run segmentation
        labels = seg.fit_transform(data, outliers)

        # Plot classification without outliers
        mm.show_classification(labels, no_outliers_data)

        # Plot classification distributions
        mm.show_classification_distributions(labels, no_outliers_data)

        # Plot classification distributions with masks
        mm.show_distributions_together(labels, no_outliers_data)  # all classes plotted together

    ## NOTE Segmentation example using persistence watershed on height property of the material
    elif example_number == "4":

        # Get outliers
        outliers = mm.extract_outliers(data, threshold=Z_SCORE_THRESH)

        # Initialize GMM segmentation
        seg = seg_water.SegmenterWatershed()

        # Choose material property to segment
        prop_data = data[:, :, 4]  # NOTE height

        # Apply merging threshold to watershed algorithm
        labels = seg.fit_transform(prop_data, outliers, 0.4)

        # Plot classification
        mm.show_classification(labels, data)

        # Plot classification distributions
        mm.show_classification_distributions(labels, data)

    ## NOTE Segmentation example removing height property after outlier removal
    elif example_number == "5":

        # Get outliers
        outliers = mm.extract_outliers(data, threshold=Z_SCORE_THRESH)
        no_outliers_data = mm.smooth_outliers_from_data(data, outliers)

        # Initialize GMM segmentation
        seg = seg_gmm.SegmenterGMM(n_components=2, embedding_dim=3)

        # Remove height property
        no_height_data = np.delete(data, 4, axis=2)

        # Run segmentation
        labels = seg.fit_transform(no_height_data, outliers)

        # Plot classification
        mm.show_classification(labels, no_outliers_data)

        # Plot classification distributions
        mm.show_classification_distributions(labels, no_outliers_data)

    ## NOTE Segmentation example with dimensionality reduction (PCA) across physical properties (skinny version of example 2)
    elif example_number == "6":

        # Initialize GMM segmentation
        seg = seg_gmm.SegmenterGMM(n_components=2, embedding_dim=3)

        # Run segmentation
        labels = seg.fit_transform(data)

        # Plot classification
        mm.show_classification(labels, data)

        # Plot classification distributions
        mm.show_classification_distributions(labels, data)

        # Plot classification distributions with masks
        mm.show_distributions_together(labels, data)  # all classes plotted together

    # NOTE Wes workflow
    elif example_number == "7":

        # Apply frequency removal
        data = mm.apply_frequency_removal(data)

        # Extract outliers
        outliers = mm.extract_outliers(data, threshold=Z_SCORE_THRESH)
        mm.show_outliers(data, outliers)
        no_outliers_data = mm.smooth_outliers_from_data(data, outliers)

        # Show a-priori distributions
        mm.show_property_distributions(data, outliers)

        # Run GMM segmentation
        seg1 = seg_gmm.SegmenterGMM(n_components=2, nonlinear=True)
        seg1_labels = seg1.fit_transform(data, outliers)

        # Show correlation after classification
        mm.show_classification_correlation(seg1_labels, no_outliers_data)

        # Remove height property (optional)
        #no_height_data = np.delete(no_outliers_data, 4, axis=2)
        #seg1_labels = seg1.fit_transform(no_height_data, outliers)

        # Plot classification
        mm.show_classification(seg1_labels, no_outliers_data)

        # Plot classification distributions
        mm.show_distributions_together(seg1_labels, no_outliers_data)

        # Create unique masks per grain
        seg2_labels = seg1.get_grains(seg1_labels)

        # Plot grain classification
        mm.show_classification(seg2_labels, no_outliers_data)

    # NOTE A-priori correlation analysis
    elif example_number == "8": # NOTE this looks better the more data files in the dir
        mm.show_correlations(data.shape[2], data_dir)

    ## NOTE Not valid example numbers
    else:
        print(f"Wrong example number inputted: {example_number}.")


if __name__ == "__main__":
    main()

