from methods import config
from methods import main_methods as mm

import numpy as np

DATA_DIR = config.data_directory_path

DATA_TYPE_SUB_DIR = "nanowires/combined/"
# DATA_TYPE_SUB_DIR = "2componentfilms/combined/"

DIR = DATA_DIR + DATA_TYPE_SUB_DIR

# Using wire films
data = np.load(DIR + "100-0_48_NW-1_combined.npy")

# Using 2 component nano films
# data = np.load(DIR + "PTB7-ThPC71BM_CBDIO_vacm_fresh_3um_combined.npy")
# data = np.load(DIR + "PTB7PC71BM_CBonly_ascast_fresh_500_nm_combined.npy")

# Shows outliers using z-score threshold
outliers = mm.extract_outliers(data)
mm.show_outliers(data, outliers)

# Shows correlations between all pairs of properties
mm.show_correlations(data.shape[2], DIR)  # TODO needs to be tested with more data files

# Shows a-priori (classification) property distributions
mm.show_property_distributions(data, outliers)

