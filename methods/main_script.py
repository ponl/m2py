from methods import config
from methods import main_methods as mm

import numpy as np

DATA_DIR = config.data_directory_path

# Using wire films
#X = np.load(DATA_DIR + "nanowires/combined/100-0_48_NW_combined.npy")

# Using 2 component nano films
X = np.load(DATA_DIR + "2componentfilms/combined/PTB7-ThPC71BM_CBDIO_vacm_fresh_3um_combined.npy")

mm.show_boundaries(X)
mm.show_outliers(X)

mm.show_feature_distributions(X)

original_x = X.copy() # TODO
