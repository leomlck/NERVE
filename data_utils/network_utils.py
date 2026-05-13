import os
import re
import numpy as np
from scipy.io import loadmat

def fetch_network_list(data_path, lbls_filename='struct_names.mat', method="p2"):
    """
    Fetches the network list and corresponding indices based on the desired parcellation strategy.

    Args:
        data_path (str): Path to the dataset directory.
        lbls_filename (str): Name of the .mat file containing structure names.
        method (str): Parcellation method to use ('p1', 'p2', or 'p3').

    Returns:
        dict: A dictionary where keys are network names and values are lists of indices.
    """
    if method == "vanilla":
        # Divide 400 ROIs into 25 contiguous blocks of 16
        network_list = {}
        for i in range(25):
            label = f'block{i}'
            indices = list(range(i * 16, (i + 1) * 16))
            network_list[label] = indices
        return network_list

    # Load structure names from the .mat file
    fc_lbls = loadmat(os.path.join(data_path, lbls_filename))['struct_names']
    fc_lbls = [s[0][0] for s in fc_lbls]

    # Extract network names based on the selected method
    if method == "p1":
        rois_lbls = [fc_lbl.split('_')[2] for fc_lbl in fc_lbls]
    elif method == "p2":
        rois_lbls = [fc_lbl.split('_')[2] + '_' + fc_lbl.split('_')[3] for fc_lbl in fc_lbls]
    elif method == "p3":
        rois_lbls = [fc_lbl.split('_')[2][:-1] for fc_lbl in fc_lbls]
    else:
        raise ValueError("Invalid method. Choose from 'p1', 'p2', or 'p3'.")

    # Create a dictionary mapping network names to their indices
    network_list = {}
    for idx, roi in enumerate(rois_lbls):
        if roi not in network_list:
            network_list[roi] = []
        network_list[roi].append(idx)

    return network_list


def shuffle_network_list(network_list, seed=None):
    """
    Shuffles the indices in a network_list while keeping the same number of ROIs
    and the same number of indices per ROI.

    Args:
        network_list (dict): Original network list {roi_name: [indices]}.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        dict: A new network list with the same ROI keys and same index counts,
              but with indices randomly shuffled across all ROIs.
    """
    if seed is not None:
        np.random.seed(seed)

    # Flatten all indices
    all_indices = np.concatenate(list(network_list.values()))
    np.random.shuffle(all_indices)

    # Create new shuffled network list
    shuffled_network_list = {}
    i = 0
    for roi, indices in network_list.items():
        count = len(indices)
        shuffled_network_list[roi] = list(all_indices[i:i + count])
        i += count

    return shuffled_network_list
