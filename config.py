# Specify hyperparameters, filepaths, and other configurations
cfg = {

    # DATA -------------------------------------------------------------------------------------------------------------
    'target': 'biomass_comps',
    # Describes the target y variable in the model can either be 'biomass_comps' or 'total_agb'
    'partition': 1,  # Take a random subset of training data to use; set as 1 to use all training data
    'in_filetype': 'las',  # Input filetype '*.las' -> not currently set up for any other file types, but could use npy
    'num_points': 5120,  # N poi to use, options: "ALL" or numbers like: 1024, 2048, 3072, 4096, 5120, 6144, 7168, ...
    'use_datasets': ["NB"],  # Possible datasets: BC, RM, PF, NB
    'use_columns': ["xyz"],  # Input features for the model
    'use_normals': True,  # Whether to use sur face normals in model (currently only supported for OCNN)

    # OCNN SPECIFIC PARAMS ----------------------------------------------------------------------------------------------
    'octree_depth': 6,  # Default 5
    'full_depth': 2,  # Default 2
    'ocnn_stages': 3,  # Number of stages used in OCNN. If "auto", is set to octree_depth - 2
    'ocnn_use_additional_features': True,
    # Boolean; Whether to use additional lidar attributes in OCNN (e.g., scan angle, intensity, etc.,)

    # PROCESSING EFFICIENCY --------------------------------------------------------------------------------------------
    'use_cudnn_autotuner': True,  # finds optimal conv https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    'use_amp': True,  # Enables automatic mixed precision training

    # FILEPATHS --------------------------------------------------------------------------------------------------------
    'pc_data_path': r'D:\Sync\RQ1\Analysis\Model_Input\train_val_test_split',
    'ref_data_path': r'D:\Sync\RQ1\Analysis\Model_Input\model_input_plot_biomass_data.csv',
    # df that contains a list of plots that were disturbed between lidar and ground sampling accoring to NTEMS
    'best_model_save_dir': rf'D:\Sync\RQ1\Analysis\Models',
    'model_checkpoint_path': r'D:\Sync\RQ1\Analysis\Model_Checkpoints\model_checkpoint.pt',
    'epoch_stats_path': r"D:\Sync\RQ1\Analysis\epoch_stats",
    'test_model_fig_out_dir': r"D:\Sync\RQ1\Analysis\Model_Test_Figures",
    'obs_and_pred_save_dir': r"D:\Sync\RQ1\Analysis\observed_and_predicted_vals_per_run",
    'ddp_train_out_pickle_path': r"D:\Sync\RQ1\Analysis\ddp_intermediate_pickles\ddp_train_output.obj",
    # Directory where train, test, and augmented lidar metrics datasets are stored
    'lidar_metrics_data_dir': r'D:\Sync\RQ1\Analysis\Model_Input\lidar_metrics',
    # Dir where rf model is saved
    'rf_model_out_dir': r'D:\Sync\RQ1\Analysis\rf_outputs\models',

    # GENERAL ----------------------------------------------------------------------------------------------------------
    'model_name': 'DGCNN',
    'loss_function_type': 'mse',  # can either be 'smooth_l1' or 'mse'
    # Specify model to use ('DGCNN' or 'OCNN_HRNet')
    'num_epochs': 200,  # Max number of epochs
    'lr': 0.001,  # Base learning rate
    'num_augs': 3,  # Number of data augmentations to apply to increase size of training set
    'batch_size': 4,
    # Number of training samples per batch to run through network (note that for DDP, this is multiplied by num GPUS)
    'n_accumulation_steps': 1,
    # Gradient accumulation: Simulated batch size = batch size * n_accumulation_steps. Turn off by setting to 1.
    'dropout_probability': 0.8,  # Dropout probability fof final mlp regressor layer
    'use_ground_points': True,  # Whether to load ground points for a point cloud
    'mlp_activation_function': 'ReLU',  # Activation function to apply through  MLP regressor

    # OPTIMIZER AND LR SCHEDULER ---------------------------------------------------------------------------------------
    'optimizer': "AdamW",  # Options: 'Adam', 'AdamW' 'SDG'
    'lr_scheduler': "CosineAnnealingWarmRestarts",
    # Options: 'OneCycle', 'CosineAnnealingWarmRestarts', 'CosineAnnealingLR', 'ReduceLROnPlateau'

    # LEARNING RATE SCHEDULER PARAMS -----------------------------------------------------------------------------------

    # OneCycleLR (From Smith & Topin, 2019, https://doi.org/10.1117/12.2520589)
    'max_lr': None,  # Maximum learning rate to be implemented in OneCycleLR. Can be estimated using lr range test.
    # Max LR suited for each architecture:
    # PointTransformer-> max_lr: 0.04

    # Cosine Annealing Warm Restart
    't_0': 20,  # Value of 10 used by (Oehmcke et al., 2021)
    't_mult': 2,  # Value of 2 used by (Oehmcke et al., 2021)

    # Cosine Annealing LR
    'eta_min': 0,  # Minimum learning rate. Default: 0

    # Reduce LR On PLateau
    'red_lr_plat_factor': 0.1,
    'red_lr_plat_patience': 5,

    # Step LR
    'step_size': 20,  # Period of learning rate decay.
    'gamma': 0.1,  # Multiplicative factor of learning rate decay.

    # OPTIMIZER PARAMS -------------------------------------------------------------------------------------------------

    # Adam Optimizer
    'adam_weight_decay': 0.01,  # Adds an L2 penalty term to loss function to help prevent overfitting
    'adam_beta_1': 0.9,  # Coefficient used for computing running averages of gradient and its square
    'adam_beta_2': 0.999,  # Coefficient used for computing running averages of gradient and its square

    # MODEL SPECIFIC PARAMS --------------------------------------------------------------------------------------------
    'DGCNN_k': 30,  # Number of Nearest Neighbors considered in DGCNN (referred to as k)

    # RF TRAINING HPs---------------------------------------------------------------------------------------------------

    # Threshold for variable importance scores generated in initial rf run with default settings, used to filter input vars
    'var_impt_thresh': 0.02,
    # The number of trees in the forest.
    'n_estimators': 100,
    # The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
    'max_depth': None,
    # The minimum number of samples required to split an internal node:
    'min_samples_split': 2,
    # The minimum number of samples required to be at a leaf node.
    'min_samples_leaf': 1,
    # Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
    'max_leaf_nodes': None,
    # The number of features to consider when looking for the best split:
    'max_features': 1.0,
    # If bootstrap is True, the number of samples to draw from X to train each base estimator.
    'max_samples': None,

    # Python DL Framework ----------------------------------------------------------------------------------------------
    'framework': 'pytorch',  # Either pytorch or pytorch geometric (pyg)

    # Misc -------------------------------------------------------------------------------------------------------------
    'save_outputs': True,
    'verbose': True,
    'experiment_note': None,

}
