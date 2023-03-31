# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
    # Codes from NVIDIA StyleGAN2-ADA (https://github.com/NVlabs/stylegan2-ada-pytorch) 
    # Project given image to the latent space of pretrained network pickle.
    # Optimize latent code, deformation code and pose simultaneously
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src import dnnlib
import math
import pdb


def project(
        G,
        c,
        pose,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        device: torch.device,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.01,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=True,
        initial_w=None,
        initial_w_deform=None,
        proj_kwargs = {},
):

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device).float()  # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_total_samples = G.mapping(torch.from_numpy(z_samples).to(device), c)  # [N, L, C]
    
    w_samples = w_total_samples['ws_orig']
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_avg_tensor = torch.from_numpy(w_avg).to(device)
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    start_w = initial_w if initial_w is not None else w_avg
    
    # Compute w_deform stats.
    logprint(f'Computing W_deform midpoint and stddev using {w_avg_samples} samples...')
    w_deform_samples = w_total_samples['ws_deform']
    w_deform_samples = w_deform_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_deform_avg = np.mean(w_deform_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_deform_avg_tensor = torch.from_numpy(w_deform_avg).to(device)
    w_deform_std = (np.sum((w_deform_samples - w_deform_avg) ** 2) / w_avg_samples) ** 0.5

    start_w_deform = initial_w_deform if initial_w_deform is not None else w_deform_avg


    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)


    w_opt = torch.tensor(start_w, dtype=torch.float32, device=device, requires_grad=True)  # pylint: disable=not-callable
    w_deform_opt = torch.tensor(start_w_deform, dtype=torch.float32, device=device, requires_grad=True)  # pylint: disable=not-callable
    lr_gamma = 0.95
    optimizer = torch.optim.Adam([w_opt] + [w_deform_opt], betas=(0.9, 0.999), lr=proj_kwargs.first_inv_lr)

    ws = {}

    for step in range(num_steps):

        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        w_deform_noise_scale = w_deform_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp


        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        w_deform_noise = torch.randn_like(w_deform_opt) * w_deform_noise_scale
        ws['ws_orig'] = (w_opt + w_noise).repeat([1, G.backbone.num_ws, 1])
        ws['ws_deform'] = (w_deform_opt + w_deform_noise).repeat([1, G.deform.num_ws, 1])
        synth_images = G.synthesis(ws, c)['image']
                

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        loss = dist


        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f} pose {pose}')

    # tmp1, tmp2 = G.backbone.num_ws, G.deform.num_ws

    del G
    return ws