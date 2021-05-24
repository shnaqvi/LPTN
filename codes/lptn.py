# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# + id="yt4rRRtlfyKL"
import sys, os
root = '/content/drive/MyDrive/Colab Notebooks/LPTN'
sys.path.append(root)

# %reload_ext autoreload
# %autoreload 2

# + id="g-tMtDLZRfJA"
import argparse
import datetime
import logging
import math
import random
import time
import torch
from os import path as osp
import sys
sys.path.append(os.getcwd())
from codes.data import create_dataloader, create_dataset
from codes.data.data_sampler import EnlargedSampler
from codes.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from codes.models import create_model
from codes.utils import (MessageLogger, check_resume,
                         get_root_logger, get_time_str, init_tb_logger,
                         init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                         set_random_seed)
from codes.utils.dist_util import get_dist_info, init_dist
from codes.utils.options import dict2str, parse


# + id="dA4zR3xjYcFH"
def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, default=os.path.join(root, 'options/train/LPTN/train_FiveK_cs221.yml'), 
                          help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args("")
    opt = parse(args.opt, is_train=is_train)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='codes', log_level=logging.INFO, log_file=log_file)
    # logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger before tensorboard logger to allow proper sync:
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loader, total_epochs, total_iters


# + colab={"base_uri": "https://localhost:8080/"} id="0r6MHfBnZ_PX" executionInfo={"elapsed": 1071, "status": "ok", "timestamp": 1621739599355, "user": {"displayName": "Salman Naqvi", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjPkTmqUm9YR_QQLazufqu2F8Nn0unp4gRPK6H6=s64", "userId": "05168687047986129733"}, "user_tz": 240} outputId="d641bbcd-2db5-4f05-c23a-cd7957d45bcc"
# parse options, set distributed setting, set ramdom seed
opt = parse_options()

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

# load resume states if necessary
if opt['path'].get('resume_state'):
    device_id = torch.cuda.current_device()
    resume_state = torch.load(
        opt['path']['resume_state'],
        map_location=lambda storage, loc: storage.cuda(device_id))
else:
    resume_state = None

# # mkdir for experiments and logger
if resume_state is None:
    make_exp_dirs(opt)
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
            'name'] and opt['rank'] == 0:
        mkdir_and_rename(osp.join('tb_logger', opt['name']))

# initialize loggers
logger, tb_logger = init_loggers(opt)

# + colab={"base_uri": "https://localhost:8080/"} id="5MRXeCW8fFP5" executionInfo={"elapsed": 3227, "status": "ok", "timestamp": 1621739610812, "user": {"displayName": "Salman Naqvi", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjPkTmqUm9YR_QQLazufqu2F8Nn0unp4gRPK6H6=s64", "userId": "05168687047986129733"}, "user_tz": 240} outputId="94991d60-3424-47c3-be52-d518ec72fe25"
# create train and validation dataloaders
result = create_train_val_dataloader(opt, logger)
train_loader, train_sampler, val_loader, total_epochs, total_iters = result

# create model
if resume_state:  # resume training
    check_resume(opt, resume_state['iter'])
    model = create_model(opt)
    model.resume_training(resume_state)  # handle optimizers and schedulers
    logger.info(f"Resuming training from epoch: {resume_state['epoch']}, "
                f"iter: {resume_state['iter']}.")
    start_epoch = resume_state['epoch']
    current_iter = resume_state['iter']
else:
    model = create_model(opt)
    start_epoch = 0
    current_iter = 0

# create message logger (formatted outputs)
msg_logger = MessageLogger(opt, current_iter, tb_logger)

# dataloader prefetcher
prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
if prefetch_mode is None or prefetch_mode == 'cpu':
    prefetcher = CPUPrefetcher(train_loader)
elif prefetch_mode == 'cuda':
    prefetcher = CUDAPrefetcher(train_loader, opt)
    logger.info(f'Use {prefetch_mode} prefetch dataloader')
    if opt['datasets']['train'].get('pin_memory') is not True:
        raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
else:
    raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                      "Supported ones are: None, 'cuda', 'cpu'.")

# + colab={"base_uri": "https://localhost:8080/"} id="fFYzB9GVenJv" executionInfo={"elapsed": 273645, "status": "ok", "timestamp": 1621739889958, "user": {"displayName": "Salman Naqvi", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GjPkTmqUm9YR_QQLazufqu2F8Nn0unp4gRPK6H6=s64", "userId": "05168687047986129733"}, "user_tz": 240} outputId="349fe970-9287-4429-c64b-d76ea1e02d1a"
    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(
                current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                model.validation(val_loader, current_iter, tb_logger,
                                 opt['val']['save_img'])

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    consumed_time = str(
        datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        model.validation(val_loader, current_iter, tb_logger,
                         opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()
