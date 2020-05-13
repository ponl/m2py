# m2py: Materials Morphology Python Package

## Contributors

- Wesley Tatum, PhD Student at University of Washington MSE Department
- Diego Torrejon, Staff Machine Learning Engineer at BlackSky
- Patrick O’Neil, Director of Machine Learning and Artificial Intelligence at BlackSky

## Installation
`Clone this repository`
`pip install -r requirements.txt`

## Usage


## Background

Thin films of semiconducting materials will enable stretchable and flexible electronic
devices, but these thin films are currently stochastic and inconsistent in their properties and
morphologies because processing and chemical conditions influence the mixing and domain size
of the different components. By using atomic force microscopy (AFM), a cheap and quick
technique, it is possible to spatially resolve and quantify these different domains based on
differences in their mechanical properties, which are strongly correlated to their electronic
performance. For this project, a library of AFM images has been curated, which includes poly(3-
hexylthiophene) that has been processed in different ways (e.g. annealing time and temperature,
thin film vs nanowire), as well as thin film mixtures of PTB7-th and PC 71 BM. To analyze these
samples, several semantic segmentation methods from the fields of machine learning and
topological data analysis are employed. Among these, a Gaussian mixture model utilizing
machine learned local geometric features proved effective. From the segmentation, probability
distributions describing the mechanical properties of each semantic segment can be obtained,
allowing the accurate classification of the various phase domains present in each sample.
