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

        # Plot classification distributions
        mm.show_classification_distributions(labels, no_outliers_data) # uses data without outliers

    ## NOTE Segmentation example with dimensionality reduction (PCA) across physical properties
    elif example_number == "2":

        # Initialize GMM segmentation
        seg = seg_gmm.SegmenterGMM(n_components=2, embedding_dim=3)

        # Run segmentation
        labels = seg.fit_transform(data)

        # Plot classification
        mm.show_classification(labels, data)

        # Plot classification distributions
        mm.show_classification_distributions(labels, data)

        # Initialize Watershed segmentation # NOTE this module can be applied after any example
        seg2 = seg_water.SegmenterWatershed(pers_thresh=2)

        # Apply watershed segmentation on output of GMM segmentation
        labels2 = seg2.fit_transform(labels)

        # Plot watershed classification
        mm.show_classification(labels2, data) # TODO why do all grains have same labels after watershed?

        # Plot watershed classification distribution
        mm.show_classification_distributions(labels2, data)

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

    ## NOTE Not valid example numbers
    else:
        print(f'Wrong example number inputted: {example_number}.')

if __name__ == '__main__':
    main()

