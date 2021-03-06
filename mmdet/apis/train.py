from __future__ import division
import os
import re
import numpy as np
from collections import OrderedDict
import glob
import shutil

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, Runner, obj_from_dict

from mmdet import datasets
from mmdet.core import (CocoDistEvalmAPHook, CocoDistEvalRecallHook, KaggleEvalHook,
                        DistEvalmAPHook, DistOptimizerHook, Fp16OptimizerHook)
from mmdet.datasets import DATASETS, build_dataloader
from mmdet.models import RPN
from .env import get_root_logger

from configs.htc.htc_hrnetv2p_w48_20e_kaggle_pku_no_semantic_translation_wudi import work_dir,\
                                                                                ds_dir, resume_from


def parse_losses(losses):

    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
       
        if isinstance(loss_value, torch.Tensor):
            if 'car_cls' not in loss_name:
                log_vars[loss_name] = loss_value.mean()
            
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError('{} is not a tensor or list of tensors'.format(loss_name))
        
    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
 
    """
    Records losses to txt file
    (used to plot trainig loss curve in Tensorboard)
    """
    files_list = glob.glob('{}*'.format(work_dir))
    current_dir = max(files_list,key=os.path.getctime)
    
    all_epoch_losses = '{}/all_epoch_losses.log'.format(current_dir)
    epoch_loss_log = '{}/epoch_losses.log'.format(current_dir)

    all_transl_losses = '{}/all_transl_losses.log'.format(current_dir)
    transl_loss_log = '{}/transl_losses.log'.format(current_dir)

    all_rot_losses = '{}/all_rot_losses.log'.format(current_dir)
    rot_loss_log = '{}/rot_losses.log'.format(current_dir)
    
    # Moves previous losses to new directory if training is accidentally interrupted
    if resume_from:
        prev_dir = os.path.dirname(resume_from)
        shutil.copy('{}/epoch_losses.log'.format(prev_dir),current_dir)
        shutil.copy('{}/transl_losses.log'.format(prev_dir),current_dir)
        shutil.copy('{}/rot_losses.log'.format(prev_dir),current_dir)
        shutil.copy('{}/mAP_log.txt'.format(prev_dir),current_dir)


    # training loss
    with open(all_epoch_losses,'a') as file:
        file.write('{}'.format(loss))
        file.write("\n")
    file.close()

    # translation loss
    with open(all_transl_losses,'a') as file:
        file.write('{}'.format(log_vars['kaggle/s2.loss_translation'].item()))
        file.write("\n")
    file.close()

    # rotation loss
    with open(all_rot_losses,'a') as file:
        file.write('{}'.format(log_vars['kaggle/s2.loss_quaternion'].item()))
        file.write("\n")
    file.close()

    num_epoch_losses = open(all_epoch_losses,'r').readlines()
    num_transl_losses = open(all_transl_losses,'r').readlines()
    num_rot_losses = open(all_rot_losses,'r').readlines()

    num_train = len(os.listdir('{}/train/'.format(ds_dir)))
    
    if len(num_epoch_losses) >= num_train:
        if len(num_epoch_losses) % num_train == 0:
            num_epochs = int((num_train*((len(num_epoch_losses) / num_train)-1)))
            
            all_ep_loss = [float(x) for x in num_epoch_losses[num_epochs:]]
            avg_epoch_loss = np.mean(all_ep_loss)
            
            all_tr_loss = [float(x) for x in num_transl_losses[num_epochs:]]
            avg_transl_loss = np.mean(all_tr_loss)

            all_ro_loss = [float(x) for x in num_rot_losses[num_epochs:]]
            avg_rot_loss = np.mean(all_ro_loss)
            
            with open(epoch_loss_log,'a') as file:
                file.write('{}'.format(avg_epoch_loss))
                file.write("\n")
            file.close()

            with open(transl_loss_log,'a') as file:
                file.write('{}'.format(avg_transl_loss))
                file.write("\n")
            file.close()

            with open(rot_loss_log,'a') as file:
                file.write('{}'.format(avg_rot_loss))
                file.write("\n")
            file.close()


    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def batch_processor(model, data, current_lr=0.001, train_mode="train"):
    losses = model(**data)
    loss, log_vars = parse_losses(losses)
    log_vars['current_lr'] = current_lr
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

    return outputs


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, validate=validate)
    else:
        _non_dist_train(model, dataset, cfg, validate=validate)


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001)
        >>> optimizer = build_optimizer(model, optimizer_cfg)
    """
    if hasattr(model, 'module'):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, torch.optim,
                             dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg['lr']
        base_wd = optimizer_cfg.get('weight_decay', None)
        # weight_decay must be explicitly specified if mult is specified
        if ('bias_decay_mult' in paramwise_options
                or 'norm_decay_mult' in paramwise_options):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get('bias_lr_mult', 1.)
        bias_decay_mult = paramwise_options.get('bias_decay_mult', 1.)
        norm_decay_mult = paramwise_options.get('norm_decay_mult', 1.)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r'(bn|gn)(\d+)?.(weight|bias)', name):
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith('.bias'):
                param_group['lr'] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group['weight_decay'] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)


def _dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds, cfg.data.imgs_per_gpu, cfg.data.workers_per_gpu, dist=True)
        for ds in dataset
    ]
    # put model on gpus
    model = MMDistributedDataParallel(model.cuda())

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(**cfg.optimizer_config,
                                             **fp16_cfg)
    else:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(DistSamplerSeedHook())
    # register eval hooks
    if validate:
        val_dataset_cfg = cfg.data.val
        eval_cfg = cfg.evaluation
        if isinstance(model.module, RPN):
            # TODO: implement recall hooks for other datasets
            runner.register_hook(
                CocoDistEvalRecallHook(val_dataset_cfg, **eval_cfg))
        else:
            if isinstance(val_dataset_cfg, dict):
                runner.register_hook(KaggleEvalHook(val_dataset_cfg, **eval_cfg))
            elif isinstance(val_dataset_cfg, list):
                for vdc in val_dataset_cfg:
                    runner.register_hook(KaggleEvalHook(vdc, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False) for ds in dataset
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    # register eval hooks
    if validate:
        val_dataset_cfg = cfg.data.val
        eval_cfg = cfg.evaluation
        if isinstance(model.module, RPN):
            runner.register_hook(CocoDistEvalRecallHook(val_dataset_cfg, **eval_cfg))
        else:
            if isinstance(val_dataset_cfg, dict):
                runner.register_hook(KaggleEvalHook(val_dataset_cfg, **eval_cfg))
            elif isinstance(val_dataset_cfg, list):
                for vdc in val_dataset_cfg:
                    runner.register_hook(KaggleEvalHook(vdc, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
