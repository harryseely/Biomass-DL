# Some parameter ranges based on this blog: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

# Modules
import optuna
from datetime import datetime as dt
import joblib

# Pre-built classes and functions
from random_forest.train_rf import train_rf

# Load global config dict
from config import cfg


# Define obective function used in optuna
def objective(trial):
    # Set the static config vars
    cfg['model_name'] = "RF"
    cfg['loss_function_type'] = "mse"
    cfg['target'] = "biomass_comps"

    # Number of augmentations
    cfg['num_augs'] = trial.suggest_int('num_augs', 0, 5)
    # Threshold for variable importance scores generated in initial rf run with default settings, used to filter input vars
    cfg['var_impt_thresh'] = trial.suggest_uniform('var_impt_thresh', 0, 0.1)
    # The number of trees in the forest.
    cfg['n_estimators'] = trial.suggest_categorical('n_estimators', [100, 200, 300, 1000, 3000])
    # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    cfg['max_depth'] = trial.suggest_categorical('max_depth', [80, 90, 100, 110])
    # The minimum number of samples required to split an internal node:
    cfg['min_samples_split'] = trial.suggest_categorical('min_samples_split', [8, 10, 12, 16])
    # The minimum number of samples required to be at a leaf node.
    cfg['min_samples_leaf'] = trial.suggest_categorical('min_samples_leaf', [3, 4, 5])
    # The number of features to consider when looking for the best split:
    cfg['max_features'] = trial.suggest_categorical('max_features',
                                                    [1, 2, 3, 'sqrt', 'log2', None])  # None means use all
    # If bootstrap is True, the number of samples to draw from X to train each base estimator.
    cfg['max_samples'] = trial.suggest_uniform('max_samples', 0.5, 1)

    # Training loop
    val_loss = train_rf(cfg, datetime="", hp_tuning_mode=True, y_vars=['foliage_z', 'branch_z', 'bark_z', 'wood_z'])

    return val_loss


# HP TUNING SECTION ----------------------------------------------------------------------------------------------------

def run_study():
    datetime = dt.now().strftime("%Y_%m_%d_%H_%M")

    # Run the study
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.RandomSampler(),
                                study_name=f"RF_random_search_{datetime}")
    study.optimize(objective, n_trials=None, timeout=3600 * 12)

    # Grab the results and export
    trials_df = study.trials_dataframe()
    trials_df.rename(columns=lambda x: x.replace('params_', '', 1))
    trials_df.to_csv(fr"E:\Optuna_Studies\optuna_study_rf_{datetime}.csv")

    # Save the study
    joblib.dump(study, fr"E:\Optuna_Studies\optuna_study_rf_study_{datetime}.pkl")


if __name__ == '__main__':
    run_study()
