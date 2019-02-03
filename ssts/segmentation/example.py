import argparse
import numpy as np
import segmentation_gmm as seg_gmm
import segmentation_watershed as seg_water
from methods import main_methods as mm # NOTE SSTS directory must be in PYTHONPATH

def main():
    # print command line arguments
    p = argparse.ArgumentParser(description="""A script with examples on how to use segmentation code.""")
    p.add_argument('data_path', help='Path to data.')
    p.add_argument('example_number', help='Specifies which example to run.')

    args = p.parse_args()
    example_number = args.example_number
    data_dir = args.data_path
    data = np.load(data_dir)

    ## NOTE Segmentation example without using outliers
    if example_number == "0":

        # Initialize GMM segmenter
        seg = seg_gmm.SegmenterGMM(n_components=2)

        # Run segmentation
        labels = seg.fit_transform(data)

        # Plot classification
        mm.show_classification(labels, data)

        # Plot classification distributions
        mm.show_classification_distributions(labels, data)

    ## NOTE Segmentation example using outliers
    elif example_number == "1":

        # Initialize GMM segmentation
        seg = seg_gmm.SegmenterGMM(n_components=2)

        # Get outliers
        outliers = mm.extract_outliers(data)

        # Run segmentation
        labels = seg.fit_transform(data, outliers)

        # Plot classification
        no_outliers_data = np.copy(data)
        no_outliers_data[outliers==1] = 0 # remove outliers from data
        mm.show_classification(labels, no_outliers_data) # uses data without outliers

        # Plot classification distribution
        mm.show_classification_distributions(labels, no_outliers_data)

        # Plot classification distributions with masks
        mm.show_distributions_together(labels, data) # all classes plotted together
        mm.show_distributions_separately(labels, no_outliers_data) # each class per plot

    ## NOTE Segmentation example with dimensionality reduction (PCA) across physical properties
    elif example_number == "2":

        # Initialize GMM segmentation
        num_pca_components = 3
        pre_seg = seg_gmm.SegmenterGMM(n_components=2, embedding_dim=num_pca_components)

        # Run segmentation
        pre_labels = pre_seg.fit_transform(data)

        # Plot classification
        mm.show_classification(pre_labels, data)

        # Plot classification distributions
        mm.show_classification_distributions(pre_labels, data)

        # Handling PCA components from data
        h, w, c = data.shape
        n = h * w
        pca_components = pre_seg.pca.transform(data.reshape(n, c)) # NOTE PCA was trained while fitting pre_seg
        pca_components = pca_components.reshape(h, w, num_pca_components) # shape (512, 512, num_pca_components)

        # Initialize Watershed segmentation # NOTE this module can be applied after any example
        post_seg = seg_water.SegmenterWatershed(pers_thresh=0.3) # TODO why is it stochastic?

        # Apply watershed segmentation on output of GMM segmentation
        post_labels = post_seg.fit_transform(pre_labels)

        # Plot watershed classification
        mm.show_classification(post_labels, data)

        # Plot area distribution of grains
        mm.show_grain_area_distribution(post_labels) # all grains plotted together

        # Plot distributions of segmented grains
        mm.show_distributions_together(post_labels, data) # all grains plotted together
        mm.show_distributions_separately(post_labels, data) # each grain per plot

    ## NOTE Segmentation example with dimensionality reduction (PCA) across neighboring pixels and physical properties
    elif example_number == "3":

        # Initialize GMM segmentation # TODO need to optimize these parameters
        seg = seg_gmm.SegmenterGMM(n_components=3, padding=3, embedding_dim=10, zscale=True)

        # Run segmentation
        labels = seg.fit_transform(data)

        # Plot classification
        mm.show_classification(labels, data)

        # Plot classification distributions
        mm.show_classification_distributions(labels, data)

    ## NOTE Segmentation example using persistence watershed on the height property of the material
    elif example_number == "4":

        # Initialize GMM segmentation
        seg = seg_water.SegmenterWatershed()

        # Choose material property to segment
        prop_data = data[:,:,4] # NOTE height

        # Run segmentation
        labels = seg.fit_transform(prop_data)

        # Plot classification
        mm.show_classification(labels, data)

        # Plot classification distributions
        mm.show_classification_distributions(labels, data)

    ## NOTE Segmentation example using persistence watershed on the output of the sobel operator
    elif example_number == "5":

        # Show boundaries after applying Sobel operator
        sobel_data = mm.show_boundaries(data)

        # Choose material property to segment
        prop_data = sobel_data[:,:,4] # NOTE height

        # Initialize GMM segmentation
        seg = seg_water.SegmenterWatershed(pers_thresh=2)

        # Run segmentation
        labels = seg.fit_transform(prop_data)

        # Plot classification
        mm.show_classification(labels, data)

        # Plot classification distributions
        mm.show_classification_distributions(labels, data)

    ## NOTE Segmentation example removing height property after outlier removal
    elif example_number == "6":

        # Initialize GMM segmentation
        pre_seg = seg_gmm.SegmenterGMM(n_components=2, embedding_dim=3)

        # Get outliers
        outliers = mm.extract_outliers(data)

        # Remove height property
        no_height_data = np.delete(data, 4, axis=2)

        # Run segmentation
        pre_labels = pre_seg.fit_transform(no_height_data, outliers)

        # Plot classification
        mm.show_classification(pre_labels, data)

        # Plot classification distributions
        mm.show_classification_distributions(pre_labels, data)

    ## NOTE Not valid example numbers
    else:
        print(f'Wrong example number inputted: {example_number}.')

if __name__ == '__main__':
    main()

