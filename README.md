# m2py: Materials Morphology Python Package

## Contributors

- Wesley Tatum, PhD Student at University of Washington MSE Department
- Diego Torrejon, Staff Machine Learning Engineer at BlackSky
- Patrick O’Neil, Director of Machine Learning and Artificial Intelligence at BlackSky

## Requirements

You need Python 3.6 or later to work with m2py.

## Installation

### Through github:
`Clone or download this repository`

`pip install -r requirements.txt`

### Through pip:
`pip install m2py`

## Usage
At the moment, we are offering two introductory tutorials found under the `tutorials` directory. There is a basic tutorial showing the main processing capabilities of m2py. There's also an advanced tutorial showing how users can use current m2py capabilities to make them suitable for their own applications.

## Background

SPM techniques have been pivotal in understanding surfaces, morphologies, and intermolecular interactions of matter from the atomic to millimeter scales. Probes interacting with the material surface produce 2-dimensional images of the topography with intensity scales representing any number of material properties (e.g. modulus, conductivity, or capacitance). In many experiments, multiple properties can be imaged simultaneously, producing a 3-dimensional stack of these images.

By utilizing different combinations of computer vision and machine learning techniques, m2py leverages differences in imaged material properties to recognize different material phases, as well as different domains and topographical features. First, outlier pixels or signals can be removed. Next, signals are compressed to only the most informative features via PCA. These signals can be deconvoluted via GMM, or some other semantic segmenter for phase labeling. Finally, an instance segmenter, such as Persistence Watershed Segmentation, clusters the phase-labeled pixels into domains, which receive another label.

Once labeled by m2py segmenters, quantitative descriptions of each domain and the total morphology are extracted and can be used to train supervised models to predict material properties or performance. Such supervised training has not been accessible to most SPM researchers, due to the labor-intensive nature of manual-labeling.
