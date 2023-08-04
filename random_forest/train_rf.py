# Modules
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from time import time
import os
from utils.z_score_tools import update_cfg_for_z_score_conversion

# My scripts
from random_forest.test_rf import test_rf


def train_rf(cfg,
             datetime,
             y_vars,
             hp_tuning_mode=False
             ):
    t0 = time()

    ##########################
    # READ TRAIN AND TEST DATA
    ##########################

    #Get list of lidar metrics to be used
    metrics_to_use = ["lad_max", "lad_mean", "lad_cv", "lad_min", "kde_peaks_count",
                      "ziqr", "zMADmean", "zMADmedian", "CRR", "zentropy", "VCI", "vn", "vFRall", "vFRcanopy", "vzrumple",
                      "vzsd", "vzcv", "OpenGapSpace", "ClosedGapSpace", "Euphotic", "Oligophotic", "HOME", "pzabovemean",
                      "pzabove2", "pzabove5", "pFirst", "pIntermediate", "pLast", "pSingle", "pMultiple", "zmax", "zmin",
                      "zmean", "zsd", "zcv", "zskew", "zkurt", "rumple", "zq1", "zq5", "zq10", "zq15", "zq20", "zq25",
                      "zq30", "zq35", "zq40", "zq45", "zq50", "zq55", "zq60", "zq65", "zq70", "zq75", "zq80", "zq85",
                      "zq90", "zq95", "zq99", "pz_below_0.15", "pz_below_2.0", "pz_below_5.0", "pz_below_10.0",
                      "pz_below_20.0", "pz_below_30.0", "zpcum1", "zpcum2", "zpcum3", "zpcum4", "zpcum5", "zpcum6",
                      "zpcum7", "zpcum8", "zpcum9", "nPoints", "maxScanAngle", "p_first_veg_returns",
                      "d10", "d20", "d30", "d40", "d50", "d60", "d70", "d80", "d90"]

    # Load train and test datasets
    train_data = pd.read_csv(os.path.join(cfg['lidar_metrics_data_dir'], 'train_lidar_metrics.csv'))
    test_data = pd.read_csv(os.path.join(cfg['lidar_metrics_data_dir'], 'test_lidar_metrics.csv'))
    val_data = pd.read_csv(os.path.join(cfg['lidar_metrics_data_dir'], 'val_lidar_metrics.csv'))

    # Get length of test data + original (not augmented) train set
    n_train = train_data.shape[0]
    n_test = test_data.shape[0]
    n_val = val_data.shape[0]

    # Partition training data if need be
    if cfg['partition'] is not None:
        train_data = train_data.sample(frac=cfg['partition'])

    #If modelling component biomass, update z-score conversion values (means and sds for each comp)
    if cfg['target'] == "biomass_comps":
        cfg.update(update_cfg_for_z_score_conversion(cfg))

    ###################
    # DATA AUGMENTATION
    ###################

    # Add pre-computed augmentations to train dataset (if specified)
    if cfg['num_augs'] > 0:
        for i in range(cfg['num_augs']):
            # Specify filepath to augmentation i
            aug_i_path = os.path.join(cfg['lidar_metrics_data_dir'], f'train_lidar_metrics_aug{i}.csv')
            # Read augmentation i
            aug_i = pd.read_csv(aug_i_path)
            # Partition augmented data if need be
            if cfg['partition'] is not None:
                aug_i = aug_i.sample(frac=cfg['partition'])
            # Add aug_i to train df
            train_data = pd.concat([train_data.reset_index(drop=True), aug_i.reset_index(drop=True)], axis=0,
                                   ignore_index=False)

        # record length of augmented train set
        n_train_aug = train_data.shape[0]

        if hp_tuning_mode is False:
            print(
                f"Adding {cfg['num_augs']} augmentations of original {n_train} for a total of {n_train_aug} training samples.")

    else:
        # If no augmentation applied, length of aug set is 0
        n_train_aug = 0

        if hp_tuning_mode is False:
            print(f"No data augmentation applied to train set.")

    #######################
    # SELECT LIDAR METRICS
    #######################

    # subset df to specific lidar metrics selected for use in model
    # Include PlotID at end of list to ensure it is included when subsetting lidar metrics to be used in training
    metrics_to_use.append("PlotID")
    train_data = train_data[metrics_to_use]
    test_data = test_data[metrics_to_use]
    val_data = val_data[metrics_to_use]

    ######################
    # READ BIOMASS REF DATA
    ######################
    ref_df = pd.read_csv(cfg['ref_data_path'], sep=",", header=0)
    ref_df = ref_df[['PlotID'] + y_vars]
    train_data = train_data.merge(ref_df, how="left", on="PlotID")
    test_data = test_data.merge(ref_df, how="left", on="PlotID")
    val_data = val_data.merge(ref_df, how="left", on="PlotID")

    ####################
    # VARIABLE SELECTION
    ####################
    if hp_tuning_mode is False:
        print(f"\n    Performing RF Variable Selection")

    # Specify y_vars and input features and ensure these and other cols are excluded from input predictors
    excluded_cols = y_vars + ['PlotID', 'las_fpath']
    features = [col for col in train_data.columns if col not in excluded_cols]

    # Define random forest classifier using default settings
    model = RandomForestRegressor(n_jobs=-1, verbose=0)

    model.fit(X=train_data[features], y=train_data[y_vars])

    # Get feature names
    feat_names = list(model.feature_names_in_)
    # Get feature importance from trained rf model
    importances = model.feature_importances_
    # Convert to pandas series for filtering
    feature_importance = pd.Series(importances, index=feat_names)
    # Sort in descending order of importance
    feature_importance = feature_importance.sort_values(ascending=False)
    # Subset to features that have an importance score greater than specified threshold
    features = list(feature_importance.loc[lambda x: x > cfg['var_impt_thresh']].index)

    ###################
    # TRAIN RF MODEL
    ###################

    if hp_tuning_mode is False:
        print(f"\n    Training RF model...")

    # Create a new random forest object
    model = RandomForestRegressor(
        n_estimators=cfg['n_estimators'],
        max_depth=cfg['max_depth'],
        min_samples_split=cfg['min_samples_split'],
        min_samples_leaf=cfg['min_samples_leaf'],
        max_leaf_nodes=cfg['max_leaf_nodes'],
        max_features=cfg['max_features'],
        max_samples=cfg['max_samples'],
        random_state=66,
        verbose=0,
        n_jobs=-1,
    )

    # fit the regressor with x and y data
    model.fit(X=train_data[features], y=train_data[y_vars])

    # Add feature names and target as an attributes to model object
    model.feature_names = features

    # Save model in compressed joblib format
    if hp_tuning_mode is False:
        joblib.dump(model, cfg['model_filepath'], compress=3)

    # Evaluate model using test dataset (use val dataset when HP tuning)
    if hp_tuning_mode:
        eval_data = val_data
    else:
        eval_data = test_data

    metrics_df, test_loss = test_rf(cfg,
                                    eval_data,
                                    y_vars=y_vars,
                                    features=features,
                                    model=model,
                                    target_label='Mg_ha',
                                    datetime=datetime,
                                    hp_tuning_mode=hp_tuning_mode
                                    )

    if hp_tuning_mode is False:
        # Record runtime
        end_time = time()
        runtime = round((end_time - t0) / 3600, 4)  # Runtime given in hours
        print(f"RF training time: {runtime} hours ({runtime * 60} minutes)")

        train_output = {
            'n_train': n_train,
            'n_test': n_test,
            'n_val': n_val,
            'n_train_aug': n_train_aug,
            'features': features,
        }

        print(metrics_df)

        return metrics_df, runtime, train_output

    else:
        return test_loss
