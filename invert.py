"""Reconstruct images and shapes from real images using pretrained network pickle."""

import os
import pdb
import importlib
import re
import yaml

import click
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src import dnnlib
from src import legacy
from src.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from src.torch_utils import misc
from reconstruction.inverter import Inverter
from gen_samples import create_samples

import configs
from omegaconf import DictConfig
import yaml


@click.command()
@click.option('--cfg_path', help='config path', type=str, required=True)


def invert_images(**kwargs):
    opts = configs.walk_configs(kwargs['cfg_path'])
    cfg = dnnlib.EasyDict(opts)
    
    network_pkl = cfg.from_args.network_pkl
    
    outdir = os.path.join(cfg.from_args.outdir, cfg.from_args.network_name)
    network_dir = os.path.join(cfg.from_args.network_dir, cfg.from_args.network_name)
    target_image_path = os.path.join(cfg.from_args.target_image_path, cfg.from_args.dataset_name)+"/"
    coach_kwargs = cfg.coach_kwargs
    
    device = torch.device('cuda')
    rank = 0
    num_gpus = 1
    random_seed = 7
    batch_size = 1
    
    
    # Load networks
    outdir = f"{outdir}/{network_pkl.replace('.pkl','')}"
    network_pkl = f"{network_dir}/{network_pkl}"
    # postfix = f'_rot{rot_angle:.3f}'
    print('Loading networks from "%s"...' % network_pkl)
    print('Save results to "%s"...' % outdir)
    with dnnlib.util.open_url(network_pkl) as f:
        resume_data = legacy.load_network_pkl(f)
        print(f'Resuming from "{network_pkl}"')
        G = resume_data['G_ema'].to(device)
        P = resume_data['P'].to(device)
    
    # Load configuration
    option_file_name = 'training_options.yaml'
    with open(os.path.join(network_dir, option_file_name), 'r') as file:
        training_options = yaml.load(file, Loader=yaml.Loader)
    data_paths = training_options.training_set_kwargs.hparams.path
    training_set_kwargs = training_options.training_set_kwargs
    data_loader_kwargs = training_options.data_loader_kwargs
    
    # Reload modules
    if cfg.from_args.reload_modules:
        print("Reloading Modules!")
        generator_name = training_options.G_kwargs.class_name.split(".")
        pkg_name, generator_name = generator_name[:-1], generator_name[-1]
        pkg_name = ".".join(pkg_name)
        generator = importlib.import_module(pkg_name)
        generator = getattr(generator, generator_name)

        G_new = generator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
    
    # Load training set.
    print('Loading training set...')
    training_set_kwargs.hparams.path = [target_image_path]
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs) # subclass of training.dataset.Dataset
    data_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=False, **data_loader_kwargs)
    
        
    os.makedirs(outdir, exist_ok=True)
    
    inverter = Inverter(data_loader,
                        G,
                        P,
                        device,
                        outdir,
                        **coach_kwargs)
    
    new_G, pivots = inverter.invert(use_projected_w=False)
    
    new_G.requires_grad_(False)
    ckpt_dit = os.path.join(outdir, 'ckpts')
    os.makedirs(ckpt_dit, exist_ok=True)
    torch.save(new_G, os.path.join(ckpt_dit, 'G.pth'))
    
    for i in range(len(pivots)):
        ws_pivot, c_pivot = pivots[f'{i}']
        torch.save(ws_pivot, os.path.join(ckpt_dit, f"ws_{i}.pt"))
        torch.save(c_pivot, os.path.join(ckpt_dit, f"c_{i}.pt"))
    
        if cfg.from_args.shapes:
            # For extracting 3D shapes
            from shape_utils import convert_sdf_samples_to_obj
            shape_res = cfg.from_args.shape_res
            max_batch=1000000
            samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=new_G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
            samples = samples.to(device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
            transformed_ray_directions_expanded[..., -1] = -1
            head = 0
            with tqdm(total = samples.shape[1]) as pbar:
                with torch.no_grad():
                    while head < samples.shape[1]:
                        torch.manual_seed(0)
                        sigma = new_G.sample_mixed(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], ws_pivot, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                        sigmas[:, head:head+max_batch] = sigma
                        head += max_batch
                        pbar.update(max_batch)

            sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
            sigmas = np.flip(sigmas, 0)

            # Trim the border of the extracted cube
            pad = int(30 * shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value
            
            os.makedirs(os.path.join(outdir, 'shape'), exist_ok=True)
            convert_sdf_samples_to_obj(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, 'shape', f'{i:04d}.obj'), level=10)


        
    
    print(f"DONE.. Check results, a new checkpoint and pivots in {outdir}")
    
    
#----------------------------------------------------------------------------
if __name__ == "__main__":
    
    invert_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------