# Python Modules
import os.path
import torch
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
from math import sqrt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn

# Pre-built classes and functions
from utils.get_model import get_model
from utils.train_dl import forward_pass, prepare_data
from utils.z_score_tools import re_convert_to_Mg_ha

def view_and_save_fig(cfg, file_desc):
    fig_export = plt.gcf()
    plt.show()
    plot_filepath = os.path.join(cfg['test_model_fig_out_dir'], f'{file_desc}.png')
    if os.path.isfile(plot_filepath):
        os.remove(plot_filepath)
    if cfg['save_outputs']:
        fig_export.savefig(plot_filepath)


def set_matching_axes(df, ax, x, y, resid_plot=False, buffer=5):
    all_vals = pd.concat((df[x], df[y]))
    ax.set_xlim([0, all_vals.max() + buffer])
    if resid_plot:
        y_abs = abs(df[y])
        y_max = max(y_abs) + buffer
        y_min = -abs(y_max)
        ax.set_ylim([y_min, y_max])
    else:
        ax.set_ylim([all_vals.min(), all_vals.max() + buffer])

    # Ensure x-axis does not become negative
    ax.set_xlim(left=0.)


def config_subplot_axis(df, metrics_df, target, comp, ax, x_axis, y_axis, resid_plot=False):
    x_vals = df[f"{comp}_{target}_{x_axis}"]
    y_vals = df[f"{comp}_{target}_{y_axis}"]
    ax.scatter(x_vals, y_vals, alpha=0.8, edgecolors='none', s=30)

    set_matching_axes(df, ax=ax, x=f"{comp}_{target}_obs", y=f"{comp}_{target}_{y_axis}", resid_plot=resid_plot)
    ax.text(0.1, 0.9,
            f"R2: {metrics_df.loc[f'{comp}_{target}', 'r2']}\nRMSE: {metrics_df.loc[f'{comp}_{target}', 'rmse']}\nMAPE: {str(round(metrics_df.loc[f'{comp}_{target}', 'mape'], 2))}",
            horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    ax.title.set_text(comp.capitalize())


def convert_array_to_df(arr, plot_id, target_label, cfg):
    """
    Function converts an input numpy array to pandas df. Ensure shape of input array matches critria described below.
    Also calculates total AGB from components and includes calculation of residuals.
    :param arr: input array of shape (obs, pred). If predicting biomass comps, col order must be bark, branch, foliage, wood
    :param plot_id: list of plot ids that match with each obs/pred row in input arr
    :param target_label: target label (e.g., Mg_ha)
    :return: 
    """
    if cfg['target'] == "total_agb":  # Fill component
        df = pd.DataFrame(arr, columns=[f'tree_{target_label}_obs', f'tree_{target_label}_pred'], index=plot_id)
        df[f"tree_{target_label}_resid"] = df[f"tree_{target_label}_obs"] - df[f"tree_{target_label}_pred"]

    elif cfg['target'] == "biomass_comps":
        # Convert to data frame
        df = pd.DataFrame(arr,
                          columns=[f'bark_{target_label}_obs', f'branch_{target_label}_obs', f'foliage_{target_label}_obs',
                                   f'wood_{target_label}_obs',
                                   f'bark_{target_label}_pred', f'branch_{target_label}_pred', f'foliage_{target_label}_pred',
                                   f'wood_{target_label}_pred'],
                          index=plot_id)

        # Add observed/predicted total biomass columns to df
        df[f"tree_{target_label}_obs"] = df[f"bark_{target_label}_obs"] + df[f"branch_{target_label}_obs"] + df[f"foliage_{target_label}_obs"] + \
                                   df[f"wood_{target_label}_obs"]
        df[f"tree_{target_label}_pred"] = df[f"bark_{target_label}_pred"] + df[f"branch_{target_label}_pred"] + df[
            f"foliage_{target_label}_pred"] + df[f"wood_{target_label}_pred"]

        # Get residuals
        df[f"tree_{target_label}_resid"] = df[f"tree_{target_label}_obs"] - df[f"tree_{target_label}_pred"]
        df[f"bark_{target_label}_resid"] = df[f"bark_{target_label}_obs"] - df[f"bark_{target_label}_pred"]
        df[f"branch_{target_label}_resid"] = df[f"branch_{target_label}_obs"] - df[f"branch_{target_label}_pred"]
        df[f"foliage_{target_label}_resid"] = df[f"foliage_{target_label}_obs"] - df[f"foliage_{target_label}_pred"]
        df[f"wood_{target_label}_resid"] = df[f"wood_{target_label}_obs"] - df[f"wood_{target_label}_pred"]
    else:
        raise Exception(f"Target: {cfg['target']} is not supported")
    
    return df


def get_metrics_make_plots(cfg, df, target_label, model_name, datetime, axis_lab, generate_plots=True):
    if cfg['model_name'] == "RF":
        cfg['num_points'] = "all"

    # Calculate test metrics for each component ------------------------------------------------------------------------

    # Create a data frame to store component metrics
    metrics_df = pd.DataFrame(columns=["r2", "rmse", "mape"],
                              index=[f"wood_{target_label}", f"bark_{target_label}", f"branch_{target_label}", f"foliage_{target_label}",
                                     f"tree_{target_label}"])

    if cfg['target'] == 'biomass_comps':
        comp_list = metrics_df.index.tolist()
    elif cfg['target'] == "total_agb":
        comp_list = [f"tree_{target_label}"]
    else:
        raise Exception(f"target_label: {cfg['target']} is not supported")

    # Loop through each biomass component get model performance metrics
    for comp in comp_list:
        metrics_df.loc[comp, "r2"] = round(metrics.r2_score(y_true=df[f"{comp}_obs"], y_pred=df[f"{comp}_pred"]), 4)
        metrics_df.loc[comp, "rmse"] = round(
            sqrt(metrics.mean_squared_error(y_true=df[f"{comp}_obs"], y_pred=df[f"{comp}_pred"])), 4)
        #Convert output proportion mape to percentage
        metrics_df.loc[comp, "mape"] = round(
            metrics.mean_absolute_percentage_error(y_true=df[f"{comp}_obs"], y_pred=df[f"{comp}_pred"])*100, 4)

    if generate_plots:
        print(metrics_df)

        # Plot total AGB biomass obs. vs. predicted  -----------------------------------------------------------------------

        # Add dataset col
        df[f"dataset"] = "blank"

        # Add a column to df for dataset
        for id in df.index.tolist():
            df.loc[id, "dataset"] = id[0:2]

        # Create plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(df[f"tree_{target_label}_obs"], df[f"tree_{target_label}_pred"],
                   alpha=0.8, edgecolors='none', s=30)

        # Set axis labels
        ax.set_xlabel("Observed Tree AGB (Mg/ha)")
        ax.set_ylabel("Predicted Tree AGB (Mg/ha)")

        plt.figtext(0.05, 0.9,
                    f"R2: {metrics_df.loc[f'tree_{target_label}', 'r2']}\nRMSE: {metrics_df.loc[f'tree_{target_label}', 'rmse']}\nMAPE: {str(round(metrics_df.loc[f'tree_{target_label}', 'mape'], 2))}",
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=ax.transAxes)

        # Add title
        plt.title("Total Tree AGB Observed vs Predicted", fontdict=None, loc='center', fontsize=15)

        set_matching_axes(df, ax, x=f"tree_{target_label}_obs", y=f"tree_{target_label}_pred")

        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

        view_and_save_fig(cfg,
                          file_desc=f"tree_{target_label}_obs_vs_pred_{cfg['num_points']}_points_{cfg['num_augs']}_augs_{model_name}_{datetime}")

        # Make residuals vs. fitted values plot for total AGB --------------------------------------------------------------
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(df[f"tree_{target_label}_pred"], df[f"tree_{target_label}_resid"],
                   alpha=0.8, edgecolors='none', s=30)

        plt.axhline(y=0, color='black', linestyle='--')

        # Add title
        plt.title("Total Tree AGB Residuals", fontdict=None, loc='center', fontsize=15)

        set_matching_axes(df, ax, x=f"tree_{target_label}_obs", y=f"tree_{target_label}_resid", resid_plot=True)

        # Set axis labels
        ax.set_xlabel("Observed Tree AGB (Mg/ha)")
        ax.set_ylabel("Residuals Tree AGB (Mg/ha)")

        view_and_save_fig(cfg,
                          file_desc=f'tree_{target_label}_residuals_{cfg["num_points"]}_points_{cfg["num_augs"]}_augs_{model_name}_{datetime}')
        if cfg['target'] == "biomass_comps":
            # Make subplots fir biomass component obs. vs. predicted   ---------------------------------------------------------
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            # Add the main title
            fig.suptitle("Component Biomass Observed vs Predicted", fontsize=15)

            config_subplot_axis(df, metrics_df, target_label, comp="bark", ax=ax[0, 0], x_axis="obs", y_axis="pred")
            config_subplot_axis(df, metrics_df, target_label, comp="branch", ax=ax[1, 0], x_axis="obs", y_axis="pred")
            config_subplot_axis(df, metrics_df, target_label, comp="foliage", ax=ax[0, 1], x_axis="obs", y_axis="pred")
            config_subplot_axis(df, metrics_df, target_label, comp="wood", ax=ax[1, 1], x_axis="obs", y_axis="pred")

            # Add axis labels
            for axis in ax.flat:
                axis.set(xlabel=f"Observed {axis_lab}", ylabel=f"Predicted {axis_lab}")
                axis.plot(axis.get_xlim(), axis.get_ylim(), ls="--", c=".3")

            # set the spacing between subplots
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.3,
                                hspace=0.3)

            view_and_save_fig(cfg,
                              file_desc=f'component_obs_vs_pred_{cfg["num_points"]}_points_{cfg["num_augs"]}_augs_{model_name}_{datetime}')

            # Make subplots for component biomass residuals --------------------------------------------------------------------
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            # Add the main title
            fig.suptitle("Component Biomass Residuals", fontsize=15)

            config_subplot_axis(df, metrics_df, target_label, comp="bark", ax=ax[0, 0], x_axis="pred", y_axis="resid",
                                resid_plot=True)
            config_subplot_axis(df, metrics_df, target_label, comp="branch", ax=ax[1, 0], x_axis="pred", y_axis="resid",
                                resid_plot=True)
            config_subplot_axis(df, metrics_df, target_label, comp="foliage", ax=ax[0, 1], x_axis="pred", y_axis="resid",
                                resid_plot=True)
            config_subplot_axis(df, metrics_df, target_label, comp="wood", ax=ax[1, 1], x_axis="pred", y_axis="resid",
                                resid_plot=True)

            # Add axis labels
            for axis in ax.flat:
                axis.set(xlabel=axis_lab, ylabel='Residuals')
                axis.axhline(y=0, c="black", linestyle='--')

            # set the spacing between subplots
            plt.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.3,
                                hspace=0.3)

            # Save plot
            view_and_save_fig(cfg,
                              file_desc=f'component_residuals_{cfg["num_points"]}_points_{cfg["num_augs"]}_augs_{model_name}_{datetime}')

    return metrics_df


def test_model(cfg,
               datetime,
               target="Mg_ha",  # Set var names for table extraction depending on target
               axis_lab="Biomass Mg/ha",
               save_obs_pred_df=True):
    # Get model name from cfg
    model_name = cfg['model_name']

    # Retrieve model architecture
    model = get_model(cfg)

    # Load mode from filepath specified in cfg
    model.load_state_dict(torch.load(cfg['model_filepath']))

    print("Testing model:", model_name)

    # Report GPU usage and set up Data Parallelization
    # Set model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if 'OCNN' not in cfg['model_name']:  # Currently, DataParallel does not seem to work for o-cnn
        # Get number of GPUs
        n_gpus = torch.cuda.device_count()
        # Report GPU usage and set up Data Parallelization
        print(f"Using {n_gpus} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Specify use dataset(s)
    use_datasets = cfg['model_filepath'].split("dataset")[1].split("useprop")[0].split("_")
    cfg['use_datasets'] = list(filter(lambda x: (x != ''), use_datasets))

    test_loader, n_test = prepare_data(cfg, ddp=False, testing=True)

    # Apply the model to test data --------------------------------------------------------------------------

    if cfg['target'] == 'biomass_comps':
        store_arr_shape = (1, 4)
    elif cfg['target'] == "total_agb":
        store_arr_shape = (1, 1)
    else:
        raise Exception(f"Target: {cfg['target']} is not supported")

    obs = np.zeros(shape=store_arr_shape, dtype=float)
    pred = np.zeros(shape=store_arr_shape, dtype=float)
    plot_id = []

    model.eval()
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_loader, colour="white", position=1, leave=False, desc=f"Testing {model_name} model"):
            # Data enters model and prediction is made
            batch_pred = forward_pass(cfg, model, batch, device)

            # Collect observed values
            batch_obs = batch['target']

            if cfg['target'] == "biomass_comps":
                # Convert from z-score to Mg/ha value
                batch_pred = re_convert_to_Mg_ha(cfg, z_components_arr=batch_pred)

                # Convert from z-score to Mg/ha value
                batch_obs = re_convert_to_Mg_ha(cfg, z_components_arr=batch_obs)

            if cfg['target'] == "total_agb":
                batch_obs = torch.reshape(batch_obs, (batch_obs.shape[0], 1))

            # Add to array of predictions
            pred = np.concatenate((pred, batch_pred.to('cpu').detach().numpy()), axis=0)

            # Add to array
            obs = np.concatenate((obs, batch_obs.detach().numpy()), axis=0)

            # Collect plotIDs
            plot_id.extend(batch['PlotID'])

    # Remove the first row from obs and pred arrays
    pred = np.delete(pred, range(0, 1), 0)
    obs = np.delete(obs, range(0, 1), 0)

    # Join arrays
    arr = np.concatenate((obs, pred), axis=1)

    df = convert_array_to_df(arr, plot_id, target, cfg)

    # Export the observed, predicted, and residuals to csv
    if cfg['save_outputs'] and save_obs_pred_df:
        df.to_csv(path_or_buf=os.path.join(cfg['obs_and_pred_save_dir'],
                                           f"obs_pred_for_{cfg['num_points']}_points_{cfg['num_augs']}_augs_{model_name}_{datetime}.csv"))

    metrics_df = get_metrics_make_plots(cfg, df, target, model_name, datetime, axis_lab)

    print(metrics_df)

    return metrics_df, df, n_test

