# Modules
import torch
from joblib import load
import numpy as np
import os

# My scripts
from utils.test_model import get_metrics_make_plots, convert_array_to_df
from utils.z_score_tools import re_convert_to_Mg_ha

# Supress warnings
import warnings

warnings.filterwarnings("ignore")


def test_rf(cfg,
            eval_data,
            y_vars,
            features=None,
            model=None,
            target_label='Mg_ha',
            axis_lab="Biomass Mg/ha",
            model_fpath=None,
            datetime="",  # Not interested in saving multiple plots with different datetimes yet
            hp_tuning_mode=False
            ):
    ##############################
    # LOAD MODEL IF PATH SPECIFIED
    ##############################
    if model is None:
        # Load model
        model = load(model_fpath)
        # Extract feature names
        features = model.feature_names_in_
        print(f"Loading model {model_fpath} and using input features \n{features}")
    else:
        if hp_tuning_mode is False:
            print(f"Testing rf model with the following features:\n{model.feature_names_in_}")

    ###############
    # TEST RF MODEL
    ###############

    # Use the forest's predict method on the test data
    pred = model.predict(eval_data[features])
    obs = eval_data[y_vars].to_numpy()

    #Adjust array shape for total AGB modelling
    if cfg['target'] == "total_agb":
        pred = np.reshape(pred, (pred.shape[0], 1))

    #Reconvert from z-score to Mg/ha if modelling biomass comps
    if cfg['target'] == "biomass_comps":
        pred = re_convert_to_Mg_ha(cfg, z_components_arr=pred)
        obs = re_convert_to_Mg_ha(cfg, z_components_arr=obs)

    # Join obs and pred arrays
    arr = np.concatenate((obs, pred), axis=1)

    # Convert to data frame
    plot_id = list(eval_data['PlotID'])
    df = convert_array_to_df(arr, plot_id, target_label, cfg)

    # Export the observed, predicted, and residuals to csv
    df.to_csv(path_or_buf=os.path.join(cfg['obs_and_pred_save_dir'],
                                       f"obs_pred_for_points_{cfg['num_augs']}_augs_{cfg['model_name']}_{datetime}.csv"))

    if hp_tuning_mode:
        generate_plots = False
    else:
        generate_plots = True

    metrics_df = get_metrics_make_plots(cfg=cfg, df=df, target_label=target_label, model_name=cfg['model_name'],
                                        datetime=datetime, axis_lab=axis_lab, generate_plots=generate_plots)

    # Get mse of each component and calculate a loss in the same manner as done for DL HP tuning
    tree_rmse_loss = metrics_df['rmse']['tree_Mg_ha']

    return metrics_df, tree_rmse_loss
