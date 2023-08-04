import torch


def get_optimizer(cfg, model):

    if cfg['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg['lr'],
                                     betas=(cfg['adam_beta_1'], cfg['adam_beta_2']),
                                     weight_decay=cfg['adam_weight_decay'])

    elif cfg['optimizer'] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cfg['lr'],
                                      betas=(cfg['adam_beta_1'], cfg['adam_beta_2']),
                                      eps=1e-08,
                                      weight_decay=cfg['adam_weight_decay'],
                                      amsgrad=False)

    elif cfg['optimizer'] == "RAdam":
        optimizer = torch.optim.RAdam(model.parameters(),
                          lr=cfg['lr'],
                          betas=(cfg['adam_beta_1'], cfg['adam_beta_2']),
                          eps=1e-08,
                          weight_decay=cfg['adam_weight_decay'])

    else:
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=cfg['lr'],
                                    momentum=0,
                                    dampening=0,
                                    weight_decay=0)

    return optimizer


def get_lr_scheduler(optimizer, cfg, train_loader):

    if cfg['lr_scheduler'] == "OneCycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=cfg['max_lr'],
                                                        epochs=cfg['num_epochs'],
                                                        steps_per_epoch=len(train_loader),
                                                        pct_start=0.3,
                                                        anneal_strategy='cos',
                                                        cycle_momentum=True,
                                                        base_momentum=0.85,
                                                        max_momentum=0.95,
                                                        div_factor=25.0,
                                                        final_div_factor=10000.0,
                                                        three_phase=False,
                                                        last_epoch=- 1,
                                                        verbose=True)

    #Cosine Annealing Warm Restart used in (Oehmcke et al., 2021) with success for point cloud DL biomass regression
    elif cfg['lr_scheduler'] == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                         T_0=cfg['t_0'], # Number of iterations for the first restart.
                                                                         T_mult=cfg['t_mult'], #A factor increases T_i after a restart.
                                                                         eta_min=0,
                                                                         last_epoch=- 1,
                                                                         verbose=False)

    #Many recent point cloud DL studies that develop new networks use CosineAnnealingLR
    # (e.g., https://arxiv.org/abs/2105.01288 ;  https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9410405)
    elif cfg['lr_scheduler'] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=cfg['num_epochs'],
                                                               eta_min=cfg['eta_min'],
                                                               last_epoch=- 1,
                                                               verbose=False)

    elif cfg['lr_scheduler'] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=cfg['red_lr_plat_factor'], # Factor by which the learning rate will be reduced. new_lr = lr * factor.
                                                               patience=cfg['red_lr_plat_patience'], # Number of epochs with no improvement after which learning rate will be reduced.
                                                               threshold=0.0001, # Threshold for measuring the new optimum, to only focus on significant changes.
                                                               threshold_mode='rel', #One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode.
                                                               cooldown=0, # Number of epochs to wait before resuming normal operation after lr has been reduced.
                                                               min_lr=0, # A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively.
                                                               eps=1e-08, # Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored.
                                                               verbose=False)

    elif cfg['lr_scheduler'] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=cfg['step_size'],
                                                    gamma=cfg['gamma'],
                                                    last_epoch=- 1,
                                                    verbose=False)

    else:
        scheduler = None

    return scheduler