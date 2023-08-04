# Modules
import torch
import torch.multiprocessing as mp
import pickle
import time
import os
from datetime import datetime as dt
import pprint as pp

# My scripts
from utils.train_dl import train_model
from utils.test_model import test_model


def run(train_model, world_size, cfg):
    # Run DDP
    mp.spawn(train_model,
             args=(world_size, cfg,),
             nprocs=world_size,
             join=True)

    # Load pickled DDP output
    train_output = pickle.load(open(cfg['ddp_train_out_pickle_path'], 'rb'))

    return train_output


def main(cfg):
    # Set general environment variables  --------------------------------------------------------------------------------

    # Set framework to DDP, so we can compare to DataParallel pytorch
    cfg['framework'] = 'pytorch_DDP'

    # Record start time and date
    start_time = time.time()
    datetime = dt.now().strftime("%Y_%m_%d_%H_%M")

    print(f"Modelling {cfg['target']}")

    # Report hyperparameters and model
    print(f"\nUsing {cfg['model_name']} architecture.")
    if cfg['verbose']:
        print("\nHyperparameters:\n")
        pp.pprint(cfg, width=1, sort_dicts=False)

    # Create model filepath where best model will be saved and updated
    model_filename = f'{cfg["model_name"]}_{dt.now().strftime("%Y_%m_%d_%H_%M")}_dataset_{"_".join(cfg["use_datasets"])}_usepropFalse.pt'
    cfg['model_filepath'] = os.path.join(cfg['best_model_save_dir'], model_filename)

    # Turn amp off for HRNet and Unet (doesn't seem to work, perhaps pytorch glitch)
    if cfg['model_name'] == "OCNN_HRNet":
        cfg['use_amp'] = False

    # Set up parallel processing and run DDP training -------------------------------------------------------------------
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    train_output = run(train_model, world_size, cfg)

    print(f"train output {train_output}")

    #Update main cfg with train output
    cfg['mean_epoch_time_seconds'] = train_output['mean_epoch_time_seconds']
    cfg['stop_epoch'] = train_output['stop_epoch']
    cfg['n_val'] = train_output['n_val']
    cfg['n_train'] = train_output['n_train']
    cfg['n_train_aug'] = train_output['n_train_aug']
    cfg['min_val_loss'] = train_output['min_val_loss']

    # Record runtime
    end_time = time.time()
    runtime = round((end_time - start_time) / 3600, 2)  # Runtime given in hours
    print(f"\nModel training time: {runtime} hours")

    # Apply the model to test data and get performance metrics ---------------------------------------------------------
    test_model(cfg, datetime)




if __name__ == "__main__":
    from config import cfg

    #For testing
    cfg['partition'] = 0.01
    cfg['num_epochs'] = 1
    cfg['save_outputs'] = True
    cfg['verbose'] = True
    cfg['num_points'] = 1024

    main(cfg)
