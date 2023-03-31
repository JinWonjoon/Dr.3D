# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import pdb
import importlib
import re
from typing import List, Optional, Tuple, Union
import json
import random
import yaml

import click
from src import dnnlib
import numpy as np
import PIL.Image
import torch
import glob
from tqdm import tqdm
import mrcfile

from src import legacy
from src.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from src.torch_utils import misc

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

@click.command()
@click.option('--json_path', 'json_path', help='json path for camera parameter', required=False)
@click.option('--network_dir', help='Network pickle filename', required=True)
@click.option('--network_pkl', help='Network pickle filename', required=False, default='')
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--rot_angle', help='rotation angle range', type=float, default=0.4)
@click.option('--rot_angle_gap', help='rotation angle gap', type=float, default=0.4)
@click.option('--force_white_back', help='render only foreground', type=bool, default=False)

def generate_images(
    json_path: str,
    network_dir: str,
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    rot_angle: float,
    rot_angle_gap: float,
    force_white_back: bool,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """
    if shapes:
        # # For extracting 3D shapes
        from shape_utils import convert_sdf_samples_to_ply

    if json_path is not None:
        with open(json_path, 'r') as f:
            labels = json.load(f)['labels']

        labels = dict(labels)
        labels = [v for k,v in labels.items()]
        labels = np.array(labels)

    device = torch.device('cuda')
    metric = 'fid50k_full'
    if len(network_pkl) == 0:
        if os.path.exists(f"{network_dir}/best_metric-{metric}.yaml"):
            print("Use best model")
            with open(f"{network_dir}/best_metric-{metric}.yaml", 'r') as f:
                fid_dict = yaml.load(f, Loader=yaml.Loader)
            if metric in fid_dict.keys():
                fid_dict = fid_dict[metric]
            for k in fid_dict.keys():
                if k[:7] == 'network':
                    network_pkl = k
                    break

        else:
            print("Use last model")
            pkl_files = glob.glob(f"{network_dir}/*.pkl")
            pkl_files.sort()
            network_pkl = pkl_files[-1].split("/")[-1]

    ## check resave
    network_pkl_resave = network_pkl.replace(".pkl", "_resave.pkl")
    if os.path.exists(f"{network_dir}/{network_pkl_resave}"):
        network_pkl = network_pkl_resave

    outdir = f"{outdir}/{network_pkl.replace('.pkl','')}"
    network_pkl = f"{network_dir}/{network_pkl}"
    postfix = f'_rot{rot_angle:.3f}'
    print('Loading networks from "%s"...' % network_pkl)
    print('Save results to "%s"...' % outdir)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    # angle_ys = [-0.4, 0., 0.4]

    angle_ys = [ang for ang in np.arange(0, rot_angle+1e-10, rot_angle_gap)]
    angle_ys.extend([ang * -1 for ang in angle_ys[1:]])
    angle_ys.sort()

    option_file_name = 'training_options_resave.yaml' if network_pkl.split("_")[-1] == "resave.pkl" else 'training_options.yaml'
    with open(os.path.join(network_dir, option_file_name), 'r') as file:
        training_options = yaml.load(file, Loader=yaml.Loader)
    data_paths = training_options.training_set_kwargs.hparams.path

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
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
    
    if force_white_back:
        G.rendering_kwargs['white_back'] = True
        if hasattr(G, 'mapping_bg'):
            G.additional_kwargs['bg_kwargs']['render_bg'] = False

    os.makedirs(outdir, exist_ok=True)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    # Generate images.
    imgss = []
    depthss = []
    for seed_idx, seed in enumerate(seeds):
        imgs = []
        depths = []
        angle_p = 0.
        cam_pivot = torch.tensor([0, 0, 0], device=device)
        cam_radius = 2.7

        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        mapping_inputs = {'z': z}
        synthesis_inputs = {}

        print(f'Generating image for seed {seed} ({seed_idx}/{len(seeds)}) ...')

        if json_path is not None:
            #use input camera pose for generator's mapping network
            idx = random.randint(0, len(labels) - 1)
            conditioning_cam2world_pose = labels[idx][:16]
            conditioning_cam2world_pose = torch.from_numpy(conditioning_cam2world_pose).to(device)
            conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).float()
            mapping_inputs.update({'c': conditioning_params})
            ws = G.mapping(**mapping_inputs, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            synthesis_inputs.update({'ws': ws, 'c': conditioning_params})
            img = G.synthesis(**synthesis_inputs)['image']
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs.append(img)

            for angle_y in angle_ys:
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                synthesis_inputs.update({'c': camera_params})
                output = G.synthesis(**synthesis_inputs)

                img = (output['image'].permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                depth = -output['image_depth'].permute(0, 2, 3, 1)
                imgs.append(img)
                depths.append(depth)

        # use canonical pose for generator's mapping network
        cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device=device)
        conditioning_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        mapping_inputs.update({'c': conditioning_params})
        ws = G.mapping(**mapping_inputs, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        synthesis_inputs.update({'ws': ws})
        for angle_y in angle_ys:
            cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot,
                                                        radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            synthesis_inputs.update({'c': camera_params})
            output = G.synthesis(**synthesis_inputs)

            img = (output['image'].permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            depth = -output['image_depth'].permute(0, 2, 3, 1)
            imgs.append(img)
            depths.append(depth)
        
        if shapes:
            max_batch=1000000
            samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
            samples = samples.to(z.device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
            transformed_ray_directions_expanded[..., -1] = -1
            head = 0
            with tqdm(total = samples.shape[1]) as pbar:
                with torch.no_grad():
                    while head < samples.shape[1]:
                        torch.manual_seed(0)
                        sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
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
            convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, 'shape', f'seed{seed:04d}.ply'), level=10)

        imgs = torch.cat(imgs, dim=2)
        depths = torch.cat(depths, dim=2)
        imgss.append(imgs)
        depthss.append(depths)

    imgss = torch.cat(imgss, dim=1)
    depthss = torch.cat(depthss, dim=1)
    min_d = torch.min(depthss)
    max_d = torch.max(depthss)
    depthss = ((depthss - min_d) / (max_d - min_d) * 255)[0, ..., 0].to(torch.uint8).cpu().numpy()
    PIL.Image.fromarray(imgss[0].cpu().numpy(), 'RGB').save(f'{outdir}/image{postfix}.png')
    PIL.Image.fromarray(depthss).save(f'{outdir}/depth{postfix}.png')

#----------------------------------------------------------------------------
if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
