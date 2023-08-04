# Other Modules
from datetime import datetime as dt
import pprint as pp
import os

# My scripts
from random_forest.train_rf import train_rf


def main(cfg):
    # Report hyperparameters
    print("\nHyperparameters:\n")
    pp.pprint(cfg, width=1, sort_dicts=False)

    # Get datetime
    datetime = dt.now().strftime("%Y_%m_%d_%H_%M")

    # Create model filepath where best model will be saved and updated
    model_filename = f'{cfg["model_name"]}_{dt.now().strftime("%Y_%m_%d_%H_%M")}_dataset_{"_".join(cfg["use_datasets"])}.pt'
    cfg['model_filepath'] = os.path.join(cfg['rf_model_out_dir'], model_filename)

    # Set target var
    if cfg['target'] == "biomass_comps":
        #y_vars = ['bark_Mg_ha', 'branch_Mg_ha', 'foliage_Mg_ha', 'wood_Mg_ha']
        y_vars = ['foliage_z', 'branch_z', 'bark_z', 'wood_z']

    elif cfg['target'] == "total_agb":
        y_vars = ['total_Mg_ha']
    else:
        raise Exception(f"Target: {cfg['target']} is not supported")

    # Implement training
    train_rf(cfg, datetime, y_vars, hp_tuning_mode=False)


if __name__ == '__main__':

    from config import cfg

    cfg['model_name'] = "RF"

    # Hyperparameters
    cfg['num_augs'] = 0
    cfg['var_impt_thresh'] = 0.005
    cfg['n_estimators'] = 200
    cfg['max_depth'] = 90
    cfg['min_samples_split'] = 12
    cfg['min_samples_leaf'] = 4
    cfg['max_features'] = 1
    cfg['max_samples'] = 0.9

    cfg['target'] = "biomass_comps"

    # Run it
    main(cfg)

