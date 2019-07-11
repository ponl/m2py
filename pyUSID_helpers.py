import os
import zipfile
import math

import numpy as np
import pyUSID as pu
import pycroscopy as px
import h5py
import matplotlib.pyplot as plt

def bruker_to_hdf5(filepath):
    """
    returns an HDF5 format file
    """
    
    translator = px.io.BrukerAFMTranslator()
    h5_file = translator.translate(filepath)
    
    return h5_file

def igor_to_hdf5(filepath):
    """
    returns an HDF5 format file
    """ 
    translator = px.io.IgorIBWTranslator()
    h5_file = translator.translate(filepath)
    
    return h5_file

def txt_to_numpy(filepath):
    """
    returns a numpy array
    """
    
    X = []
    with open(filepath) as txt:
        for line in txt:
            row = [float(x.replace("\n", "")) for x in line.split("\t")]
            X.append(row)
    
    return np.array(X)

def mk_h5_file(path):

    h5_fl = h5py.File(path, mode = 'r')
    
    return h5_fl

def mk_USID_file(path):
    
    h5_fl = mk_h5_file(path)
    
    used_file = pu.USIDataset(h5_fl)
    
    return usid_file

def plot_single_channel(hdf5_channel, title):
    """
    Takes: HDF5 channel and a title (as a string) then plots that data channel
    returns: fig, axes for the plotted channel in question
    """
    
    df = np.array(hdf5_channel)
    oldx, oldy = df.shape
    x = y = int(math.sqrt(oldx))
    df = df.reshape(x, y)
    
    fig, axes = plt.subplots(ncols = 1, figsize = (6, 6))
    pu.plot_utils.plot_map(axes, df, show_xy_ticks = False, show_cbar = False)
    axes.set_title(title)
    
    return fig, axes

def plot_all_channels(hdf5_measurement_group, num_cols = 2):
    """
    This function takes in a HDF5 measurement group of spectra, identifies all main datasets, and plots them in a
    n x m grid. Default column setting is 2. Returns fig and axes of the subplot group
    """
    
    measurement_group = pu.hdf_utils.get_all_main(hdf5_measurement_group)
    num_channels = len(measurement_group)
    num_rows = 1
    title = ''
    channel_iterator = 0

    if num_channels%num_cols == 0:
        num_rows = int(num_channels / num_cols)
    else:
        num_rows = int(math.floor((num_channels / 2) + 1))

    fig, axes = plt.subplots(num_rows, num_cols, figsize = (10, (num_rows * 5)))

    for ax in axes.flat:
        channel = np.array(measurement_group[channel_iterator])

        oldx, oldy = channel.shape
        x = y = int(math.sqrt(oldx))
        channel = channel.reshape(x, y)

        title = pu.hdf_utils.get_attr(measurement_group[channel_iterator], 'quantity')
        title1 = title.split('"')[1]
        ax.matshow(channel)
        ax.set_title(title1)
        channel_iterator += 1

    plt.tight_layout()

    return fig, axes

def save_single_scan_png(input_path, output_path, channel_index):
    """
    Takes in a str for a directory to an hdf5 files, outgoing path for .PNG file, index of the desired channel
    Returns: none, writes file to desired path
    """
    
    h5_fl = mk_h5_file(input_path)
    
    channel_list = pu.io.hdf_utils.get_all_main(h5_fl)
    
    chnl = channel_list[channel_index]
    title = pu.io.hdf_utils.get_attr(chnl, 'quantity')

    plot_single_channel(chnl, title)
    plt.savefig(output_path, format = 'png')
    
    return

def save_all_channels_png(input_filename, output_filenames, file_source):
    """
    Takes in str and list of str for input and output file directory/name, respectively, and the source filetype
    Returns: none, writes all png files to directories/names supplied in output_filenames
    """
    
    igor_amfm_channel_list = ['HeightRetrace', 'Amplitude1Retrace', 'Phase1Retrace', 'FrequencyRetrace',
                                  'DissipationRetrace', 'YoungsRetrace', 'IndentationRetrace']
    igor_cafm_channel_list = []
    bruker_qnm_channel_list = ['Height Sensor', 'Peak Force Error', 'DMTModulus', 'LogDMTModulus',
                               'Adhesion', 'Deformation', 'Dissipation', 'Height']
    
    channel_list = []
    file_name = input_filename
    h5_file = mk_h5_file(file_name)
    channels = pu.io.hdf_utils.get_all_main(h5_file)
    
    if file_source == 'igor_amfm':
        channel_list = igor_amfm_channel_list
    elif file_source == 'igor_cafm':
        channel_list = igor_cafm_channel_list
    elif file_source == 'bruker_qnm':
        channel_list = bruker_qnm_channel_list
    else:
        for h in range(len(channels)):
            chnl = channels[h]
            channel_list.append(pu.io.hdf_utils.get_attr(chnl, 'quantity'))
    

    for i in range(len(channel_list)):
        chnl1 = channels[i]
        title = channel_list[i]
        plot_single_channel(chnl1, title)       
        plt.savefig(output_filenames[i], format = 'png')
        
    return

