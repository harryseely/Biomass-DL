# TODO: Check out this tips and tricks for DDP (includes how to integrate early stopping) https://github.com/Lance0218/Pytorch-DistributedDataParallel-Training-Tricks


# Modules
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchmetrics import R2Score
from torchmetrics import MeanAbsolutePercentageError as MAPE
import numpy as np
import pickle
import os
import os.path
from copy import deepcopy
from time import time
from os.path import basename
import csv

# Pre-built classes and functions
from utils.get_model import get_model
from utils.optimizers_and_lr_schedulers import get_lr_scheduler
from utils.optimizers_and_lr_schedulers import get_optimizer
from utils.point_cloud_dataset import PointCloudsInFilesPreSampled, augment_data
from utils.ocnn_custom_utils import CustomCollateBatch
from utils.z_score_tools import re_convert_to_Mg_ha
from utils.z_score_tools import update_cfg_for_z_score_conversion

# Supress warnings
import warnings


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_data(cfg, ddp=True, testing=False):

    # Load mean and sd of each biomass component so that z-scores can be converted back to Mg/ha and save in cfg
    if cfg['target'] == "biomass_comps":
        cfg.update(update_cfg_for_z_score_conversion(cfg))

    # If not using DDP, use shuffle
    if DDP:
        shuffle = False
    else:
        shuffle = True

    # Determine whether to use a batch collate function (required for octree/ocnn models)
    if 'OCNN' in cfg['model_name']:
        collate_fn = CustomCollateBatch(cfg, batch_size=cfg['batch_size'], merge_points=False)
    else:
        collate_fn = None

    if testing is False:
        # Set up train data sampler and loader
        dataset_train = PointCloudsInFilesPreSampled(cfg, set='train', partition=cfg['partition'])
        # Get length of train dataset before augmentation for records
        n_train = len(dataset_train)

        # Apply data augmentation to training dataset
        dataset_train = augment_data(cfg, train_dataset=dataset_train, verbose=cfg['verbose'])

        # Determine whether to use a sampler (only use when implementing DDP)
        if ddp == True:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        else:
            train_sampler = None
        train_loader = DataLoader(dataset_train, batch_size=cfg['batch_size'], shuffle=shuffle,
                                  num_workers=0, sampler=train_sampler, collate_fn=collate_fn,
                                  drop_last=True,
                                  pin_memory=True)

        # Set up val data sampler and loader
        dataset_val = PointCloudsInFilesPreSampled(cfg, set='val', partition=None)
        # Determine whether to use a sampler (only use when implementing DDP)
        if ddp == True:
            val_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
        else:
            val_sampler = None
        val_loader = DataLoader(dataset_val, batch_size=cfg['batch_size'], shuffle=shuffle,
                                num_workers=0, sampler=val_sampler, collate_fn=collate_fn,
                                drop_last=True,
                                pin_memory=True)

        # Record size of augmented training dataset + val datasets
        n_train_aug = len(dataset_train)
        n_val = len(dataset_val)

        return train_loader, train_sampler, val_loader, val_sampler, n_train, n_train_aug, n_val
    else:
        test_dataset = PointCloudsInFilesPreSampled(cfg, set='test', partition=None)

        test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=0,
                                 pin_memory=True, collate_fn=collate_fn)

        n_test = len(test_dataset)

        return test_loader, n_test


def write_output(rank, epoch, epoch_times, epoch_mn_val_loss_list, cfg, n_train, n_train_aug, n_val):
    # Save the model training output to pickel using 1st GPU
    if rank == 0:
        # Store the outputs in a dict and then save as pickle for subsequent use
        train_outs = {
            'stop_epoch': str(epoch),
            'mean_epoch_time_seconds': np.mean(epoch_times),
            'min_val_loss': min(epoch_mn_val_loss_list),
            'n_train': n_train,
            'n_train_aug': n_train_aug,
            'n_val': n_val,

        }

        pickle.dump(train_outs, open(cfg['ddp_train_out_pickle_path'], 'wb'))


def loss_fn(pred, y, cfg):
    """
    :param pred: predicted vals
    :param y: target vals
    :param cfg: config dict
    :return:
    """

    if cfg['loss_function_type'] == "smooth_l1":
        calc_loss = torch.nn.SmoothL1Loss()
    elif cfg['loss_function_type'] == "mse":
        calc_loss = torch.nn.MSELoss()
    else:
        raise Exception(f"{cfg['loss_function_type']} loss function is not supported")

    if cfg['target'] == "biomass_comps":
        # Compute mse loss for each component
        loss_bark = calc_loss(pred[:, 0], y[:, 0])
        loss_branch = calc_loss(pred[:, 1], y[:, 1])
        loss_foliage = calc_loss(pred[:, 2], y[:, 2])
        loss_wood = calc_loss(pred[:, 3], y[:, 3])

        #Get loss of total AGB
        tree_pred = pred[:, 0] + pred[:, 1] + pred[:, 2] + pred[:, 3]
        tree_obs = y[:, 0] + y[:, 1] + y[:, 2] + y[:, 3]
        loss_tree = calc_loss(tree_pred, tree_obs)

        # Calculate mse loss using loss for each component relative to its contribution to total biomass
        loss = loss_bark + loss_branch + loss_foliage + loss_wood + loss_tree


    elif cfg['target'] == "total_agb":

        #Ensure pred and y have same shape (issue for DGCNN)
        if pred.shape != y.shape:
            y = torch.reshape(y, pred.shape)

        loss = calc_loss(pred, y)
    else:
        raise Exception(f"Target: {cfg['target']} is not supported")


    return loss


def forward_pass(cfg, model, batch, rank):

    if 'OCNN' not in cfg['model_name']:
        input = batch['points'].to(device=rank)
    else:
        input = {k: v.to(device=rank) if hasattr(v, 'to') else v for k, v in batch.items()}

    pred = model(input)

    return pred


def train_model(rank, world_size, cfg):
    print(f"Running DDP example on rank {rank}.")
    setup(rank, world_size)

    # Set path to store epoch stats
    epoch_stats_file = basename(cfg['model_filepath']).replace('.pt', '.csv')
    epoch_stats_file = os.path.join(cfg['epoch_stats_path'], epoch_stats_file)

    # Create epoch stats csv with header
    if cfg['save_outputs']:
        headerList = ['epoch', 'train_loss', 'val_loss', 'train_r2', 'val_r2', 'train_mape', 'val_mape', 'lr_at_end_epoch']
        with open(epoch_stats_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headerList)

    # Instantiate automatic mixed precision scaling gradient scaling to avoid gradients flushing to zero (“underflowing”)
    scaler = torch.cuda.amp.GradScaler()

    # Instantiate train and val R2/MAPE score classes from torch metrics
    train_r2 = R2Score(num_outputs=4, adjusted=0, multioutput='uniform_average', dist_sync_on_step=False).to(rank)
    val_r2 = R2Score(num_outputs=4, adjusted=0, multioutput='uniform_average', dist_sync_on_step=False).to(rank)
    train_mape = MAPE(dist_sync_on_step=False).to(rank)
    val_mape = MAPE(dist_sync_on_step=False).to(rank)

    # create model and move it to GPU with id rank
    model = get_model(cfg).to(rank)
    model = DDP(model, device_ids=[rank])

    # Load datasets, apply augmentation to train data, set up samplers for DDP
    train_loader, train_sampler, val_loader, val_sampler, n_train, n_train_aug, n_val = prepare_data(cfg)

    # Set optimizer to be used in training
    optimizer = get_optimizer(cfg, model)

    # Set learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, cfg, train_loader)

    # Switch on the cuDNN Autotuner
    # Slows down training initially, but then speeds it up as CUDA has to find the best way to implement convolutions
    if cfg['num_epochs'] > 5:
        torch.backends.cudnn.benchmark = True

    # List to store epoch mean val loss and mape for early stopping
    epoch_mn_val_loss_list = []
    epoch_val_mape_list = []

    # List to store time required for each epoch
    epoch_times = []

    iters = len(train_loader)

    # Loop through each epoch
    for epoch in range(0, cfg['num_epochs']):

        #Free up GPU memory before epoch
        torch.cuda.empty_cache()

        # Record epoch start time on one GPU
        if rank == 0:
            # Get start time
            t0 = time()

        # helps reduce fragmentation of GPU memory in certain cases (https://org/docs/stable/generated/torch.cuda.empty_cache.html)
        torch.cuda.empty_cache()

        # ***************************************************************************
        # ********************************* TRAINING ********************************
        # ***************************************************************************

        # Put model in training mode
        model.train()
        # Shuffle train set
        train_sampler.set_epoch(epoch)

        # Create lists to store losses
        train_losses = []

        for step, batch in enumerate(train_loader):

            # Runs the forward pass under autocast.
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg['use_amp']):

                # Data enters model and prediction is made
                pred = forward_pass(cfg, model, batch, rank)

                if cfg['use_amp']:
                    # output is float16 because linear layers autocast to float16.
                    assert pred.dtype is torch.float16

                # Get target and send to correct device for batch parallel
                y = batch['target'].to(rank)

                # Compute loss using custom loss function
                train_loss = loss_fn(pred, y, cfg)

                # loss is float32 because mse_loss layers autocast to float32.
                assert train_loss.dtype is torch.float32

            # Normalize the Gradients
            if cfg['n_accumulation_steps'] > 1:
                train_loss = train_loss / cfg['n_accumulation_steps']
            # Backwards propagation
            scaler.scale(train_loss).backward()

            if ((step + 1) % cfg['n_accumulation_steps'] == 0) or (step + 1 == len(train_loader)):
                # Implement optimizer using the gradient scaler
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

                # Clear the gradients (note this method is faster than typical optimizer.zero_grad() )
                # According to : https://org/tutorials/recipes/recipes/tuning_guide.html
                for param in model.parameters():
                    param.grad = None

            if cfg['lr_scheduler'] == "CosineAnnealingWarmRestarts":
                scheduler.step(epoch + step / iters)

            # Convert pred and y from z-score to Mg/ha value to compute R^2
            if cfg['target'] == "biomass_comps":
                pred = re_convert_to_Mg_ha(cfg, z_components_arr=pred)
                y = re_convert_to_Mg_ha(cfg, z_components_arr=y)

            # Send train loss to cpu and update  list
            train_losses.append(train_loss.item())

            if cfg['target'] == "total_agb":
                y = torch.reshape(y, (y.shape[0], 1))

            # Update train r2/mape score
            if len(pred) > 1:
                train_r2.update(pred, y)
                train_mape.update(pred, y)
            else:
                print("WARNING: Batch prediction only includes a single sample!")

            # Report step stats
            if cfg['verbose']:
                print(
                    f"Rank: {rank} - Epoch {epoch} train loss {train_loss} progress: {int(step / len(train_loader) * 100)}% ")

        # ***************************************************************************
        # ********************************* VALIDATION ******************************
        # ***************************************************************************

        # List to store val losses (on each device)
        val_losses = []

        # Do not compute gradient for validation section
        with torch.no_grad():
            model.eval()
            # Shuffle val set
            val_sampler.set_epoch(epoch)
            # Start validation loop
            for step, batch in enumerate(val_loader):

                # Data enters model and prediction is made
                pred = forward_pass(cfg, model, batch, rank)

                # Get target and send to correct device for batch parallel
                y = batch['target'].to(rank)

                # Compute loss using custom loss function
                val_loss = loss_fn(pred, y, cfg)

                # Send to cpu and update val loss list
                val_losses.append(val_loss.item())

                # Convert pred and y from z-score to Mg/ha value to compute R^2
                if cfg['target'] == "biomass_comps":
                    pred = re_convert_to_Mg_ha(cfg, z_components_arr=pred)
                    y = re_convert_to_Mg_ha(cfg, z_components_arr=y)

                if cfg['target'] == "total_agb":
                    y = torch.reshape(y, (y.shape[0], 1))

                # Update train r2 score
                if len(pred) > 1:
                    val_r2.update(pred, y)
                    val_mape.update(pred, y)
                else:
                    print("WARNING: Batch prediction only includes a single sample!")

                if cfg['verbose']:
                    print(
                        f"Epoch {epoch} rank {rank} val loss {val_loss} progress: {int(step / len(val_loader) * 100)}% ")

        # At end of epoch, adjust the learning rate using scheduler (if not none)
        if scheduler is not None:
            if cfg['lr_scheduler'] == "ReduceLROnPlateau":
                scheduler.step(val_loss)
            elif cfg['lr_scheduler'] != "CosineAnnealingWarmRestarts":
                scheduler.step()
            else:
                pass

        # **************************** REPORT EPOCH STATS ***************************

        # Compute the epoch mean train and val r2/mape scores
        epoch_train_r2 = train_r2.compute().item()
        epoch_val_r2 = val_r2.compute().item()
        epoch_train_mape = train_mape.compute().item()
        epoch_val_mape = val_mape.compute().item()

        # Reset torch metric states after each epoch
        train_r2.reset()
        val_r2.reset()
        train_mape.reset()
        val_mape.reset()

        # Compute the mean epoch train/val losses (MSE) on each device
        epoch_mn_train_loss = np.mean(train_losses)
        epoch_mn_val_loss = np.mean(val_losses)

        # *************** Wait for all GPUs to reach this point before any proceed further **********************
        dist.barrier()

        # Record several epoch stats on first GPU (vars are synched across GPUs in code above)
        # Early stopping and model saving also performed on first GPU (below)
        if rank == 0:

            # Record epoch duration
            t1 = time()
            epoch_duration = round(t1 - t0, 1)
            epoch_times.append(epoch_duration)

            # Record learning rate and update learning rate list
            lr_at_end_epoch = optimizer.state_dict()["param_groups"][0]["lr"]

            # Report epoch stats on first GPU
            print("  | Epoch: " + str(epoch) +
                  "  | Val loss: " + str(np.round_(epoch_mn_val_loss, 2)) +
                  "  | Train loss: " + str(np.round_(epoch_mn_train_loss, 2)) +
                  "  | Val R^2: " + str(np.round_(epoch_val_r2, 2)) +
                  "  | Train R^2: " + str(np.round_(epoch_train_r2, 2)) +
                  "  | Val MAPE: " + str(np.round_(epoch_val_mape, 2)) +
                  "  | Train MAPE: " + str(np.round_(epoch_train_mape, 2)) +
                  "  | Time: " + str(epoch_duration))

            # Update epoch stats in csv
            if cfg['save_outputs']:
                with open(epoch_stats_file, 'a') as f:
                    f.write(
                        f'{epoch}, '
                        f'{epoch_mn_train_loss}, {epoch_mn_val_loss}, {epoch_train_r2}, {epoch_val_r2}, {epoch_train_mape}, {epoch_val_mape}, {lr_at_end_epoch}\n'
                    )

            # Update epoch val mape and mean loss lists
            epoch_mn_val_loss_list.append(epoch_mn_val_loss)
            epoch_val_mape_list.append(epoch_val_mape)

            # **************************** MODEL SAVING / EARLY STOPPING / OUTPUTS ***************************

            # Determine whether to save the model based on val loss and mape
            val_improvement = epoch_mn_val_loss <= min(epoch_mn_val_loss_list)
            if cfg['model_filepath'] and val_improvement:
                    print(f"Saving model for epoch: {epoch}")
                    # Get model state dict (different depending on architecture)
                    model_state = model.module.state_dict()
                    # Save best model state so far
                    best_model_state = deepcopy(model_state)
                    torch.save(best_model_state, cfg['model_filepath'])

        # Early stopping
        if cfg['early_stopping'] is True:
            raise Exception("Early stopping is not set up currently!")

    if rank == 0:
        # Using first GPU, write output to pickle
        write_output(rank, epoch, epoch_times, epoch_mn_val_loss_list, cfg, n_train, n_train_aug, n_val)

    # End DDP and dissolve multiprocessing
    cleanup()
    
    return
