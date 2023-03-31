# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks."

Code adapted from
"Alias-Free Generative Adversarial Networks"."""

from email.policy import default
import os
import click
import re
import json
import tempfile
import torch
from omegaconf import DictConfig
import yaml
import datetime
import pdb
import numpy as np

import configs
from src.loggers import wandblogger
from src import dnnlib
from src.training import training_loop
from src.metrics import metric_main
from src.torch_utils import training_stats
from src.torch_utils import custom_ops

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'
    else:
        wandb_logger = wandblogger.wandblogger(c)
        c.wandb_cfg.wandb_logger = wandb_logger
        
    # Execute training loop.
    #del c.from_args
    save_cfg_again = training_loop.training_loop(rank=rank, **c)

    if rank == 0:
        wandb_logger.finish()
        if save_cfg_again:
            save_yaml_path = f"{os.path.dirname(c['from_args']['resume'])}/training_options_resave.yaml"
            print(save_yaml_path)
            with open(save_yaml_path, 'wt') as f:
                yaml.dump(c, f, indent=2)

#----------------------------------------------------------------------------

def launch_training(cfg, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    # Pick output directory.
    # prev_run_dirs = []
    # if os.path.isdir(outdir):
    #     prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    # prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    # prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    # cur_run_id = max(prev_run_ids, default=-1) + 1

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if cfg.wandb_cfg['wandb_name'] is None:
        cfg.wandb_cfg['wandb_name'] = f'{desc}_{now}'
    else:
        cfg.wandb_cfg['wandb_name'] = f'{cfg.wandb_cfg["wandb_name"]}_{desc}_{now}'
    
    assert cfg.wandb_cfg['wandb_online'] in [True, False], "Choose whether to run wandb online(True) / offline(False) !!"

    cfg.run_dir = os.path.join(outdir, cfg.wandb_cfg['wandb_name'])
    assert not os.path.exists(cfg.run_dir)
    # Print options.
    print()
    print('Training options:')
    print(json.dumps(cfg, indent=2))
    print()
    print(f'Output directory:    {cfg.run_dir}')
    print(f'Number of GPUs:      {cfg.num_gpus}')
    print(f'Batch size:          {cfg.batch_size} images')
    print(f'Training duration:   {cfg.total_kimg} kimg')
    print(f'Dataset path:        {cfg.training_set_kwargs.hparams.path}')
    print(f'Dataset resolution:  {cfg.training_set_kwargs.hparams.resolution}')
    # print(f'Dataset labels:      {cfg.training_set_kwargs.hparams.use_labels}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(cfg.run_dir)
    # with open(os.path.join(cfg.run_dir, 'training_options.json'), 'wt') as f:
    #     json.dump(cfg, f, indent=2)
    with open(os.path.join(cfg.run_dir, 'training_options.yaml'), 'wt') as f:
         yaml.dump(cfg, f, indent=2)
    
    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if cfg.num_gpus == 1:
            subprocess_fn(rank=0, c=cfg, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(cfg, temp_dir), nprocs=cfg.num_gpus)

#----------------------------------------------------------------------------

# def init_dataset_kwargs(data):
#     try:
#         dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
#         dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
#         dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
#         dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
#         dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
#         return dataset_kwargs, dataset_obj.name
#     except IOError as err:
#         raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------
@click.command()
@click.option('--cfg_path', help='config path', type=str, required=True)
def main(**kwargs):
    """Train a GAN using the techniques described in the paper
    "Alias-Free Generative Adversarial Networks".

    Examples:

    \b
    # Train StyleGAN3-T for AFHQv2 using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan3-t --data=~/datasets/afhqv2-512x512.zip \\
        --gpus=8 --batch=32 --gamma=8.2 --mirror=1

    \b
    # Fine-tune StyleGAN3-R for MetFaces-U using 1 GPU, starting from the pre-trained FFHQ-U pickle.
    python train.py --outdir=~/training-runs --cfg=stylegan3-r --data=~/datasets/metfacesu-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=6.6 --mirror=1 --kimg=5000 --snap=5 \\
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl

    \b
    # Train StyleGAN2 for FFHQ at 1024x1024 resolution using 8 GPUs.
    python train.py --outdir=~/training-runs --cfg=stylegan2 --data=~/datasets/ffhq-1024x1024.zip \\
        --gpus=8 --batch=32 --gamma=10 --mirror=1 --aug=noaug
    """

    # Initialize config.
    opts = configs.walk_configs(kwargs['cfg_path'])
    opts['from_args']['print_angle'] = opts['from_args'].get('print_angle', False)

    # this is to retrain superresolution from 512-64 model
    opts['from_args']['resume_partial_copy'] = opts['from_args'].get('resume_partial_copy', False)
    opts['from_args']['use_balance_batch'] = opts['from_args'].get('use_balance_batch', False)
    opts['from_args']['augment_per_data'] = opts['from_args'].get('augment_per_data', {})

    opts['from_args']['freeze_generator'] = opts['from_args'].get('freeze_generator', False)
    opts['from_args']['Nofreeze_keywords'] = opts['from_args'].get('Nofreeze_keywords', [])
    if opts['from_args']['freeze_generator']:
        assert isinstance(opts['from_args']['Nofreeze_keywords'], list)
        assert len(opts['from_args']['Nofreeze_keywords']) > 0, "No modules for optimization (no modules to be freezed)"

    opts['from_args']['run_Dr3D'] = opts['from_args'].get('run_Dr3D', False)

    opts['D_kwargs']['additional_kwargs'] = opts['D_kwargs'].get('additional_kwargs', {})
    opts['G_kwargs']['additional_kwargs'] = opts['G_kwargs'].get('additional_kwargs', {})

    opts['G_kwargs']['additional_kwargs']['rot_angle'] = opts['G_kwargs']['additional_kwargs'].get('rot_angle', 2*np.pi/6)
    opts['G_kwargs']['additional_kwargs']['rot_angle_gap'] = opts['G_kwargs']['additional_kwargs'].get('rot_angle_gap', np.pi/6)
    # opts['G_kwargs']['rendering_kwargs']['avg_camera_pivot'] = [0, 0, -0.2]

    # opts['D_kwargs']['additional_kwargs']['use_mismatch_d'] = opts['D_kwargs']['additional_kwargs'].get('use_mismatch_d', False)
    # opts['G_kwargs']['additional_kwargs']['mismatch_d'] = opts['G_kwargs']['additional_kwargs'].get('mismatch_d', 0)

    # if use mismatch_d, multi-Discriminator is not allowed
    if 'D_dual_kwargs' in opts['D_kwargs']:
        assert opts['D_kwargs']['additional_kwargs']['use_mismatch_d'] == False and opts['G_kwargs']['additional_kwargs']['mismatch_d'] == 0

    cfg = dnnlib.EasyDict(opts) # Command line arguments.

    # # Hyperparameters & settings.
    cfg.G_opt_kwargs.lr = (0.002 if cfg.from_args.cfg == 'stylegan2' else 0.0025) if cfg.from_args.glr is None else cfg.from_args.glr

    # Sanity checks.
    if cfg.batch_size % cfg.num_gpus != 0:
        raise click.ClickException('--batch must be a multiple of --gpus')
    if cfg.batch_size % (cfg.num_gpus * cfg.batch_gpu) != 0:
        raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if hasattr(cfg.D_kwargs, "epilogue_kwargs"):
        # Default mnDomIN Discriminator
        if cfg.batch_gpu < cfg.D_kwargs.epilogue_kwargs.mbstd_group_size:
            raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    else:
        # Multi Discriminator
        if cfg.batch_gpu < cfg.D_kwargs.D1_kwargs.epilogue_kwargs.mbstd_group_size or cfg.batch_gpu < cfg.D_kwargs.D2_kwargs.epilogue_kwargs.mbstd_group_size:
            raise click.ClickException('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in cfg.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    if 'sr_module_class_name' in cfg.from_args.keys():
        sr_module = cfg.from_args.sr_module_class_name
    else:
        if cfg.training_set_kwargs.hparams.resolution == 512:
            sr_module = 'src.training.superresolutions.superresolution.SuperresolutionHybrid8XDC'
        elif cfg.training_set_kwargs.hparams.resolution == 256:
            sr_module = 'src.training.superresolutions.superresolution.SuperresolutionHybrid4X'
        elif cfg.training_set_kwargs.hparams.resolution == 128:
            sr_module = 'src.training.superresolutions.superresolution.SuperresolutionHybrid2X'
        else:
            assert False, f"Unsupported resolution {cfg.training_set_kwargs.hparams.resolution}; make a new superresolution module"

    cfg.G_kwargs.rendering_kwargs.superresolution_module = sr_module
    cfg.G_kwargs.rendering_kwargs.c_gen_conditioning_zero = not cfg.from_args.gen_pose_cond 
    cfg.G_kwargs.rendering_kwargs.gpc_reg_prob = cfg.from_args.gpc_reg_prob if cfg.from_args.gen_pose_cond else None

    if cfg.from_args.density_reg > 0:
        cfg.G_reg_interval = cfg.from_args.density_reg_every

    cfg.loss_kwargs.gpc_reg_prob = cfg.from_args.gpc_reg_prob if cfg.from_args.gen_pose_cond else None

    # Augmentation.
    # if opts.aug != 'noaug':
    #     cfg.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
    #     if opts.aug == 'ada':
    #         cfg.ada_target = opts.target
    #     if opts.aug == 'fixed':
    #         cfg.augment_p = opts.p

    # Resume.
    if cfg.from_args.resume is not None:
        cfg.resume_pkl = cfg.from_args.resume
        if len(opts['from_args']['augment_per_data']) > 0:
            cfg.augment_kwargs.ada_kimg = 100 # Make ADA react faster at the beginning.
        cfg.ema_rampup = None # Disable EMA rampup.
        if not cfg.from_args.resume_blur:
            cfg.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.
            cfg.loss_kwargs.gpc_reg_fade_kimg = 0 # Disable swapping rampup

    # Performance-related toggles.
    # if opts.fp32:
    #     cfg.G_kwargs.num_fp16_res = cfg.D_kwargs.num_fp16_res = 0
    #     cfg.G_kwargs.conv_clamp = cfg.D_kwargs.conv_clamp = None

    if 'backbone_kwargs' not in cfg.G_kwargs.keys():
        cfg.G_kwargs.num_fp16_res = cfg.from_args.g_num_fp16_res
        cfg.G_kwargs.conv_clamp = 256 if cfg.from_args.g_num_fp16_res > 0 else None
        cfg.D_kwargs.num_fp16_res = cfg.from_args.d_num_fp16_res
        cfg.D_kwargs.conv_clamp = 256 if cfg.from_args.d_num_fp16_res > 0 else None

    # Description string.
    desc = f'{cfg.from_args.dataset_name:s}-gpus{cfg.num_gpus:d}-batch{cfg.batch_size:d}-gamma{cfg.loss_kwargs.r1_gamma:g}'
    # Launch.
    launch_training(cfg=cfg, desc=desc, outdir=cfg.from_args.outdir, dry_run=cfg.from_args.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
