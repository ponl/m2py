from methods import config
from methods import main_methods as mm

import numpy as np

DATA_DIR = config.data_directory_path
DATA_TYPE_SUB_DIR = "nanowires/combined/"
DIR = DATA_DIR + DATA_TYPE_SUB_DIR

# Using wire films
X = np.load(DIR + "100-0_48_NW-1_combined.npy")

# Using 2 component nano films
#X = np.load(DIR + "PTB7-ThPC71BM_CBDIO_vacm_fresh_3um_combined.npy")

# Show boundaries after applying Sobel operator
mm.show_boundaries(X)

# Shows outliers using z-score threshold
mm.show_outliers(X)

# Shows correlations between all pairs of properties
mm.show_correlations(X.shape[2], DIR) # TODO needs to be tested with more data files

# Shows a-priori (classification) property distributions
mm.show_property_distributions(X)

# Shows pixel classification after applying Gaussian mixture model
L, reduced_X = mm.apply_segmentation(X, height_flag=False)
mm.show_classification(L, reduced_X)

# Shows a-posteriori (classification) property distributions
mm.show_classification_distributions(L, X)

