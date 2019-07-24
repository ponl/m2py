import os

import numpy as np


def get_all_paths(path):
    """
    Gets list of data files in path
    
    Parameters
    ----------
        path : str
            data directory
    
    Returns
    ----------
        fls : list
            data files
    """
    fls = [os.path.join(path, f) for f in os.listdir(path)]
    return fls


def get_sample_count(path):
    """
    Gets number of data files in path
    
    Parameters
    ----------
        path : str
            data directory
        
    Returns
    ----------
        int
            number of data files
    """
    return len(get_all_paths(path))


def get_path_from_index(index, path):
    """
    Gets specific data file from path
    
    Parameters
    ----------
        index : int
            specific data file index
        path : str
            data directory
    
    Returns
    ----------
        str
            data file
    """
    return get_all_paths(path)[index]


def get_data(index, path):
    """
    Loads specific data file
    
    Parameters
    ----------
        index : int
            specific data file index
        path : str
            data directory
        
    Returns
    ----------
       NumPy Array
           data from file
    """
    fl = get_path_from_index(index, path)
    return np.load(fl)
