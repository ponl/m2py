import os
import sys
from decimal import Decimal

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from methods import main_methods as mm

import segmentation_gmm as seg_gmm
import segmentation_watershed as seg_water

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

qnm_fields = ['Adhesion', 'Deformation', 'Dissipation', 'Height', 'Modulus', 'Stiffness']

def array_stats(array):
    """
    Takes in numpy array and calculates median, standard deviation, and variance
    
    Args:
    array – np.array()
    
    Returns:
    median – float64
    std_dev – float64
    var – float64
    """
    median = np.median(array)
    std_dev = np.std(array)
    var = np.var(array)
    
    return median, std_dev, var

def phase_sort(array, labels, n_components):
    """
    Takes in a 3D numpy array and sorts pixels into a dictionary based on phase labels
    
    Args:
    array – np.array() of SPM data
    phase_labels – np.array() of phase labels
    n_components – int number of phases for sorting
    
    Returns:
    phases – dictionary of numpy arrays, where each key is the phase number and each value is
        a flattened array of the pixels in that phase and their properties
    """
    x,y,z = array.shape

    data = np.reshape(array, ((x*y), z))
    labels = np.reshape(labels, (x*y))
    
    phase_list = mm.get_unique_labels(labels)
    phase_list.sort()

    phase0 = []
    phase1 = []
    phase2 = []
    phase3 = []
    phase4 = []
    phase5 = []
    phase6 = []
    phase7 = []
    phase8 = []
    phase9 = []

    for i in range(len(labels)):
        if labels[i] == 0:
            phase0.append(data[i, :])
        elif labels[i] == 1:
            phase1.append(data[i, :])
        elif labels[i] == 2:
            phase2.append(data[i, :])
        elif labels[i] == 3:
            phase3.append(data[i, :])
        elif labels[i] == 4:
            phase4.append(data[i, :])
        elif labels[i] == 5:
            phase5.append(data[i, :])
        elif labels[i] == 6:
            phase6.append(data[i, :])
        elif labels[i] == 7:
            phase7.append(data[i, :])
        elif labels[i] == 8:
            phase8.append(data[i, :])
        elif labels[i] == 9:
            phase9.append(data[i, :])
        else:
            pass
        
    phases = {}
    
    if len(phase0) >= 1:
        phases[0] = np.asarray(phase0)
    else:
        pass
    
    if len(phase1) >= 1:
        phases[1] = np.asarray(phase1)
    else:
        pass
    
    if len(phase2) >= 1:
        phases[2] = np.asarray(phase2)
    else:
        pass
    
    if len(phase3) >= 1:
        phases[3] = np.asarray(phase3)
    else:
        pass
    
    if len(phase4) >= 1:
        phases[4] = np.asarray(phase4)
    else:
        pass
    
    if len(phase5) >= 1:
        phases[5] = np.asarray(phase5)
    else:
        pass
    
    if len(phase6) >= 1:
        phases[6] = np.asarray(phase6)
    else:
        pass
    
    if len(phase7) >= 1:
        phases[7] = np.asarray(phase7)
    else:
        pass
    
    if len(phase8) >= 1:
        phases[8] = np.asarray(phase8)
    else:
        pass
    
    if len(phase9) >= 1:
        phases[9] = np.asarray(phase9)
    else:
        pass
    
    return phases

def plot_single_phase_props(array):
    """
    Takes in a flattened array of SPM data and plots the histogram distributions of a single phase of the data
    
    Args:
    array – np.array() of SPM data
    
    Returns:
    plots figures
    """
    xy,z = array.shape
    print ("Shape: ", array.shape)

    fig = plt.figure(figsize=(10,15))
    for i in range(z):
        counts, bins = np.histogram(array[:,i], bins = 30)
        ymax = counts.max()

        median, std_dev, var = array_stats(array[:,i])

        plt.subplot(3,2,1+i)
        plt.hist(array[:,i], bins = 30, alpha = 0.5)
        plt.plot([median, median], [0, ymax+5], label = 'Median', linewidth = 3, c = 'r')
        plt.plot([median+std_dev, median+std_dev], [0,ymax+5], label = 'Std Dev', linewidth = 2, c = 'k')
        plt.plot([median-std_dev, median-std_dev], [0,ymax+5], linewidth = 2, c = 'k')
        plt.title(f'{qnm_fields[i]}'+'  Median: '+ '{:.2e}'.format(median)+',   Stand. Dev.: '+'{:.2e}'.format(std_dev))
        plt.ylim(0, ymax+5)
        plt.legend()

    plt.tight_layout()
    plt.show()
    
    return


def plot_all_phases_props(phases):
    """
    Takes in a flattened array of SPM data and plots the histogram distributions of all phases of the data
    
    Args:
    phases – dictionary of numpy arrays, where each key is the phase number and each value is
        a flattened array of the pixels in that phase and their properties
    
    Returns:
    plots figures
    """
    for k, v in phases.items():
        print ('Phase ', k,)
        plot_single_phase_props(v)
    
    return

def gen_phase_stats(phases):
    """
    Takes in dictionary of phases from phase_sort() and generates a parallel dictionary of stats
    
    Args:
    phases – dictionary of numpy arrays, where each key is the phase number and each value is
        a flattened array of the pixels in that phase and their properties
        
    Returns:
    phase_stats – dictionary of numpy arrays, where each key is the phase number and each value is
        a flattened array of the basic statistical analysis of the phase properties
    """
    phase_stats = {}
    keys = ['median', 'standard deviation', 'variance']

    for k, v in phases.items():
        xy, z = v.shape
        for i in range(z):
            phase_stats[k] = {}
        for h in range(z):
            phase_stats[k][h] = array_stats(v[:,h])
            
    return phase_stats


def grain_sort(array, grain_labels):
    """
    Takes in an array and the associated domain (grain) labels. Sorts pixels into a dictionary, where each value is
    an array of the pixel values in the domain
    
    Args:
    array – np.array() of SPM data
    domain_labels – np.array() of domain labels
    n_components – int number of phases for sorting
    
    Returns:
    grain_props – dictionary of numpy arrays, where each key is the phase number and each value is
        a flattened array of the pixels in that grain and their properties
    """
    x,y,z = array.shape

    data = np.reshape(array, ((x*y), z))
    labels = np.reshape(grain_labels, (x*y))

    unique_labels = mm.get_unique_labels(labels)
    unique_labels.sort()

    grain_count = len(unique_labels)

    grain_props = {}
    for h in range(x*y):
        pixel_props = data[h,:]
        label = labels[h]

        if label in grain_props:
            old_value = np.asarray(grain_props[label])
            new_value = np.vstack((old_value, pixel_props))    # Append new pixel vector to old vectors
        else:
            new_value = np.asarray(pixel_props)

        grain_props[label] = new_value
    
    return grain_props

def gen_grain_stats(grain_props):
    """
    Takes in a dictionary of the different grains, produced by grain_sort(), and calculates basic histogram statistics
    on the constituent pixels
    
    Args:
    grain_props – dictionary of numpy arrays, where each key is the phase number and each value is
        a flattened array of the pixels in that grain and their properties
        
    Returns:
    grain_stats – dictionary of numpy arrays, where each key is the grain number and each value is
        a flattened array of the basic statistical analysis of the grain properties
    """
    grain_stats = {}
    keys = ['median', 'standard deviation', 'variance']
    
    for k,v in grain_props.items():
        grain = np.asarray(v)

        if len(grain.shape) == 1: 
            tup = grain.shape
            z = tup[0]
            for i in range(z):
                grain_stats[k] = {}
            for i in range(z):
                grain_stats[k][i] = array_stats(grain[i])
        else:
            xy, z = grain.shape
            for i in range(z):
                grain_stats[k] = {}
            
            for i in range(z):        
                grain_stats[k][i] = array_stats(grain[:,i])
            
    return grain_stats
