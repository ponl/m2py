import argparse
import numpy as np
import segmentation_gmm as seg_gmm
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
        seg = seg_gmm.SegmenterGMM(n_components=3)

        # Run segmentation
        labels = seg.fit_transform(data)

        # Plot classification
        mm.show_classification(labels, data)

        # Plot classification distributions
        mm.show_classification_distributions(labels, data)

    ## NOTE Segmentation example using outliers
    elif example_number == "1":

        # Initialize GMM segmented
        seg = seg_gmm.SegmenterGMM(n_components=3)

        # Get outliers
        outliers = mm.extract_outliers(data)

        # Run segmentation
        labels = seg.fit_transform(data, outliers)

        # Plot classification
        mm.show_classification(labels, data)

        # Plot classification distributions
        mm.show_classification_distributions(labels, data)

    ## NOTE Segmentation example with dimensionality reduction (PCA) across physical properties
    elif example_number == "2":

        # Initialize GMM segmented
        seg = seg_gmm.SegmenterGMM(n_components=2, embedding_dim=3)

        # Run segmentation
        labels = seg.fit_transform(data)

        # Plot classification
        mm.show_classification(labels, data)

        # Plot classification distributions
        mm.show_classification_distributions(labels, data)

    ## NOTE Segmentation example with dimensionality reduction (PCA) across neighboring pixels and physical properties
    elif example_number == "3":

        # Initialize GMM segmented
        seg = seg_gmm.SegmenterGMM(n_components=3, padding=3, embedding_dim=10, zscale=True)

        # Run segmentation
        labels = seg.fit_transform(data)

        # Plot classification
        mm.show_classification(labels, data)

        # Plot classification distributions
        mm.show_classification_distributions(labels, data)

    else:
        print(f'Wrong example number inputted: {example_number}.')

if __name__ == '__main__':
    main()

