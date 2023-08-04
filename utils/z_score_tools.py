# Redefine the functions used in z score tools

import pandas as pd
import numpy as np
import torch


def update_cfg_for_z_score_conversion(cfg):
    """
    Loads a csv of the reference data and gets the mean and sd for use in converting back from z score.
    Global config dictionary (cfg) is updated with these values for each biomass component for conversion during training.
    :param ref_data_path: filepath to reference data csv.
    :param cfg: global config dictionary
    :return: updated global config dictionary.
    """

    ref_data = pd.read_csv(cfg['ref_data_path'])

    # Get mean and sd for each component and update the global cfg
    for comp in ['bark', 'branch', 'foliage', 'wood']:
        cfg[f'{comp}_Mg_ha_mn'] = np.mean(ref_data[f'{comp}_Mg_ha'])
        cfg[f'{comp}_Mg_ha_sd'] = np.std(ref_data[f'{comp}_Mg_ha'])

    return cfg


def convert_from_z_score(z_vals, sd, mean):
    """
    Converts z-score back to original value using mean and sd
    :param cfg: global config dict that contains the mean and sd values needed for conversion
    :param z_vals: z-score values to be converted
    :param sd: standard deviation of original data
    :param mean: mean of original data
    :return: input values converted to back to original units
    """

    # X = Z * standard_deviation + mean
    converted_val = z_vals * sd + mean

    return converted_val


def re_convert_to_Mg_ha(cfg: dict, z_components_arr):
    """
    Converts array of component z score value back to biomass value in Mg/ha
    ***IMPORTANT: array needs to enter function with columns as follows: bark, branch, foliage, wood

    :param cfg: global config dict that contains the mean and sd values for each component needed for conversion
    :param z_components_arr: input np array of 'branch', 'bark', 'foliage', 'wood' values (in z score format)
    :return: tensor -> input values converted to Mg/ha units, note that this tensor no longer has gradients and is only for calculating performance metrics
    """

    #Send tensor to cpu if needed
    if torch.is_tensor(z_components_arr):
        converted_arr = z_components_arr.detach().clone()
    else:
        converted_arr = z_components_arr

    #Re-convert z-score to original value for each component
    for col_number, comp in zip(range(0, 4), ['bark', 'branch', 'foliage', 'wood']):
        comp_z_vals = converted_arr[:, col_number]
        converted_arr[:, col_number] = convert_from_z_score(comp_z_vals, sd=cfg[f'{comp}_Mg_ha_sd'],
                                                            mean=cfg[f'{comp}_Mg_ha_mn'])

    return converted_arr
