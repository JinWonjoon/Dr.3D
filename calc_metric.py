# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

import os
import pdb

import click
import json
import tempfile
import copy
import torch
import importlib
import yaml

from src import dnnlib
from src import legacy
from src.metrics import metric_main
from src.metrics import metric_utils
from src.torch_utils import training_stats
from src.torch_utils import custom_ops
from src.torch_utils import misc
import numpy as np

from src.training.pose_networks.hopenet import Posenet

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    """

    Args:
        rank:
        args: metrics=metrics, num_gpus=gpus, network_pkl=network_pkl, verbose=verbose,
                           rerun=rerun, postfix=postfix, Gvr_yaw_std=Gvr_yaw_std, Gmn_cam_dist_info=None
        temp_dir:

    Returns:

    """
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

    # Print network summary.
    device = torch.device('cuda', rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)
    P = copy.deepcopy(args.P).eval().requires_grad_(False).to(device)
    if rank == 0 and args.verbose:
        print('Finish Constructing networks')

    # Calculate each metric.
    for metric in args.metrics:
        if args.rerun is False and os.path.exists(os.path.join(args.run_dir, f'metric-{metric}{args.postfix}.yaml')):
            with open(os.path.join(args.run_dir, f'metric-{metric}{args.postfix}.yaml'), 'r') as f:
                prev_dict_pkl = yaml.load(f, Loader=yaml.Loader)
            network_pkl = args.network_pkl.split("/")[-1]
            
            if network_pkl in prev_dict_pkl.keys():
                if rank == 0 and args.verbose:
                    print(f"{metric}{args.postfix} is already calculated")
                continue

        if rank == 0 and args.verbose:
            print(f'Calculating  {metric}{args.postfix}...')
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
        if rank == 0 and metric[:3] == 'fid':
            save_genimg_dir = f"{args.run_dir}/gen_images"
            os.makedirs(save_genimg_dir, exist_ok=True)
            pkl_num = args.network_pkl.split('/')[-1].replace('.pkl','')
            save_genimg_name = f"{save_genimg_dir}/{pkl_num}_{metric}{args.postfix}.png"
        else:
            save_genimg_name = None
        result_dict = metric_main.calc_metric(metric=metric, G=G, P=P, dataset_kwargs=args.dataset_kwargs,
                                                num_gpus=args.num_gpus, rank=rank, device=device, progress=progress,
                                                save_genimg_name=save_genimg_name, Gvr_yaw_std=args.Gvr_yaw_std, Gmn_cam_dist_info=args.Gmn_cam_dist_info)
        if rank == 0:
            #metric_main.report_metric(result_dict, run_dir=args.run_dir, snapshot_pkl=args.network_pkl)
            metric_main.report_metric_per_pkl(result_dict, run_dir=args.run_dir, snapshot_pkl=args.network_pkl, postfix=args.postfix)
        if rank == 0 and args.verbose:
            print()

    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')

#----------------------------------------------------------------------------

class CommaSeparatedList(click.ParamType):
    name = 'list'

    def convert(self, value, param, ctx):
        _ = param, ctx
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

#----------------------------------------------------------------------------

def calc_metrics(ctx, network_pkl, metrics, gpus, verbose, find_best, rerun, Gvr_yaw_std, Gmn, cal_best):
    """Calculate quality metrics for previous training run or pretrained network pickle.

    Examples:

    \b
    # Previous training run: look up options automatically, save result to JSONL file.
    python calc_metrics.py --metrics=pr50k3_full \\
        --network=~/training-runs/00000-ffhq10k-res64-auto1/network-snapshot-000000.pkl

    \b
    # Pre-trained network pickle: specify dataset explicitly, print result to stdout.
    python calc_metrics.py --metrics=fid50k_full --data=~/datasets/ffhq.zip --mirror=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

    Available metrics:

    \b
      ADA paper:
        fid50k_full  Frechet inception distance against the full dataset.
        kid50k_full  Kernel inception distance against the full dataset.
        pr50k3_full  Precision and recall againt the full dataset.
        is50k        Inception score for CIFAR-10.

    \b
      StyleGAN and StyleGAN2 papers:
        fid50k       Frechet inception distance against 50k real images.
        kid50k       Kernel inception distance against 50k real images.
        pr50k3       Precision and recall against 50k real images.
        ppl2_wend    Perceptual path length in W at path endpoints against full image.
        ppl_zfull    Perceptual path length in Z for full paths against cropped image.
        ppl_wfull    Perceptual path length in W for full paths against cropped image.
        ppl_zend     Perceptual path length in Z at path endpoints against cropped image.
        ppl_wend     Perceptual path length in W at path endpoints against cropped image.
    """

    # Validate arguments.
    postfix = "" if Gvr_yaw_std == -1 and Gmn == 'est' else f"_Gmn{Gmn}_Gvryawstd{Gvr_yaw_std}"
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, network_pkl=network_pkl, verbose=verbose,
                           rerun=rerun, postfix=postfix, Gvr_yaw_std=Gvr_yaw_std, Gmn_cam_dist_info=None)

    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        ctx.fail('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        ctx.fail('--gpus must be at least 1')
        ## find best model : fid only!!

    assert os.path.isfile(network_pkl)
    pkl_dir = os.path.dirname(network_pkl)
    args.run_dir = pkl_dir
    os.makedirs(args.run_dir, exist_ok=True)

    assert os.path.isfile(os.path.join(pkl_dir, 'training_options.yaml'))
    args.run_dir = pkl_dir
    print(f"save dir : {args.run_dir}")

    with open(os.path.join(pkl_dir, 'training_options.yaml'), 'r') as file:
        training_options = yaml.load(file, Loader=yaml.Loader)

    dataset_class = training_options.training_set_kwargs.class_name

    if find_best:
        data_paths = training_options.training_set_kwargs.hparams.path
        print("find best model")
        for metric in args.metrics:

            if metric[:3] == 'fid':
                with open(os.path.join(args.run_dir, f'metric-{metric}{postfix}.yaml'), 'r') as f:
                    dicts = yaml.load(f, Loader=yaml.Loader)
                best = dicts['best']
                pkl_names = [pkl_name for pkl_name in dicts.keys() if '.pkl' in pkl_name]

                try:    ## old version
                    values = [(float(dicts[pkl_name][0]['results'][metric]), pkl_name) for pkl_name in pkl_names]
                except:
                    values = [(float(dicts[pkl_name][0][0]['results'][metric]), pkl_name) for pkl_name in pkl_names]
                
                values.sort()

                if best == 'min':
                    best_value, best_pkl_name = values[0]
                elif best == 'max':
                    best_value, best_pkl_name = values[-1]

                best_method = f"_{cal_best}" if cal_best is not None else ''
                best_metric_file_path = os.path.join(args.run_dir, f'best_metric-{metric}{best_method}.yaml')

                if os.path.exists(best_metric_file_path):
                    with open(best_metric_file_path, 'r') as f:
                        best_dicts = yaml.load(f, Loader=yaml.Loader)
                    if 'best' in best_dicts.keys():
                        best_dicts = {}
                else:
                    best_dicts = {}
                best_dict = {
                    f"{metric}{postfix}": {
                    'metric': f"{metric}{postfix}",
                    'best': best,
                    'pkl_range': pkl_names,
                    best_pkl_name: dicts[best_pkl_name]
                    }
                }
                best_dicts.update(best_dict)
                with open(best_metric_file_path, 'w') as f:
                    yaml.dump(best_dicts, f, indent=2)
                print(best_dicts)
    else:
        data_paths = training_options.training_set_kwargs.hparams.path
        
        if Gmn == 'prior':
            try:
                args.Gmn_cam_dist_info = training_options.G_kwargs.additional_kwargs.cam_dist_info
            except:
                args.Gmn_cam_dist_info = training_options.loss_kwargs.additional_kwargs.cam_dist_info

        # Load network.
        if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
            ctx.fail('--network must point to a file or URL')
        if args.verbose:
            print(f'Loading network from "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl, verbose=args.verbose) as f:
            network_dict = legacy.load_network_pkl(f)
            args.G = network_dict['G_ema'] # subclass of torch.nn.Module
            # args.P = Posenet(pretrained_path=training_options.from_args['posenet_path']).eval().requires_grad_(False)
            args.P = Posenet(pretrained_path="./ckpts/hopenet_ckpt.pth").eval().requires_grad_(False)

        # Initialize dataset options.
        if network_dict['training_set_kwargs'] is not None:
            args.dataset_kwargs = dnnlib.EasyDict(network_dict['training_set_kwargs'])
            # For not separating dataset in normal and steep clusters.
            args.dataset_kwargs.hparams.path = training_options.training_set_kwargs.hparams.path
            # args.dataset_kwargs.hparams.json_name = training_options.training_set_kwargs.hparams.json_name
        else:
            ctx.fail('Could not look up dataset options; please specify --data')

        # Finalize dataset options.
        # args.dataset_kwargs.hparams.json_name = os.path.join(os.path.dirname(data), json_name)
        # Print dataset options.
        if args.verbose:
            print('Dataset options:')
            print(json.dumps(args.dataset_kwargs, indent=2))

        # Locate run dir.
        # args.run_dir = None

        # Launch processes.
        if args.verbose:
            print('Launching processes...')
        torch.multiprocessing.set_start_method('spawn')
        with tempfile.TemporaryDirectory() as temp_dir:
            if args.num_gpus == 1:
                subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
            else:
                torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)


@click.command()
@click.pass_context
@click.option('network_pkl', '--network', help='Network pickle filename or URL', metavar='PATH', required=True)
@click.option('--metrics', help='Comma-separated list or "none"', type=CommaSeparatedList(), default='fid50k_full', show_default=True)
# @click.option('--data', help='Dataset to evaluate metrics against (directory or zip) [default: same as training data]', metavar='PATH')
# @click.option('--json_name', help='json file for metadata [default: same as training data]', metavar='PATH')
# @click.option('--data_pose', help='Dataset to evaluate metrics against (directory or zip) [default: same as training data]', metavar='PATH')
# @click.option('--mirror', help='Whether the dataset was augmented with x-flips during training [default: look up]', type=bool, metavar='BOOL')
@click.option('--gpus', help='Number of GPUs to use', type=int, default=1, metavar='INT', show_default=True)
@click.option('--verbose', help='Print optional information', type=bool, default=True, metavar='BOOL', show_default=True)
#@click.option('--run_dir', help='where to save', required=False, type=str, default = None, metavar='PATH')
@click.option('--pkl_range', help='(min, max) pkl range : include min, max value', required=False, type=CommaSeparatedList(), default=None)
@click.option('--find_best', help='find best model only', type=bool, default=False, metavar='BOOL', show_default=True)
@click.option('--rerun', help='rerun calculation', type=bool, default=False, metavar='BOOL')
@click.option('--Gvr_yaw_std', help='std(degree) of prior distribution for sampling epsilon_Gvr. if -1, then sampling from data', type=int, default=-1)
@click.option('--Gmn', help='cam dist info for Gmn : estimation (pose-estimation network) or prior', type=str, default='est')
@click.option('--cal_best', help='how to calculate metric to find best model. if None, mean. if number, use fid.', type=int, default=None)
def main(ctx, network_pkl, metrics, gpus, verbose, pkl_range, find_best, rerun, gvr_yaw_std, gmn, cal_best):
    dnnlib.util.Logger(should_flush=True)
    assert gmn in ['est', 'prior']
    if find_best:
        calc_metrics(ctx, network_pkl, metrics, gpus, verbose, find_best, rerun, gvr_yaw_std, gmn, cal_best)
    else:
        assert pkl_range is not None
        assert len(pkl_range) in [1, 2]
        if len(pkl_range) == 1:
            pkl_range.extend(pkl_range)
        pkl_range = [int(i) for i in pkl_range]
        pkl_range.sort()
        if "_resave" in network_pkl:
            return
        network_num = int(network_pkl.split("/")[-1].replace("network-snapshot-", "").replace(".pkl", ""))
        if pkl_range[0] <= network_num <= pkl_range[1]:
            calc_metrics(ctx, network_pkl, metrics, gpus, verbose, False, rerun, gvr_yaw_std, gmn, cal_best)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------
