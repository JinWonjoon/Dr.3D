# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import pdb

import torch
import numpy as np
from src.torch_utils import persistence
from src.camera_utils import compute_angle_from_matrix, LookAtPoseSampler
from src import dnnlib

@persistence.persistent_class
class TriPlaneGeneratorV1(torch.nn.Module):
    def __init__(self,
            z_dim,  # Input latent (Z) dimensionality.
            c_dim,  # Conditioning label (C) dimensionality.
            w_dim,  # Intermediate latent (W) dimensionality.
            img_resolution,  # Output resolution.
            img_channels,  # Number of output color channels.
            sr_num_fp16_res=0,
            rendering_kwargs={},
            sr_kwargs={},
            backbone_kwargs={},
            decoder_kwargs={},
            additional_kwargs={}
    ):
        super().__init__()
        self.additional_kwargs = additional_kwargs
        self.z_dim = z_dim
        self.c_dim = 0 if rendering_kwargs['c_gen_conditioning_zero'] else c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.renderer = dnnlib.util.construct_class_by_name(**additional_kwargs['renderer']) #ImportanceRenderer()
        self.ray_sampler = dnnlib.util.construct_class_by_name(**additional_kwargs['ray_sampler']) #RaySampler()

        backbone_kwargs['c_dim'] = self.c_dim
        self.backbone = dnnlib.util.construct_class_by_name(**backbone_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(**sr_kwargs)
        self.decoder = dnnlib.util.construct_class_by_name(**decoder_kwargs)
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        self._last_planes = None

        rot_angle = additional_kwargs['rot_angle']
        rot_angle_gap = additional_kwargs['rot_angle_gap']
        angle_ys = [ang for ang in np.arange(0, rot_angle+1e-10, rot_angle_gap)]
        angle_ys.extend([ang * -1 for ang in angle_ys[1:]])
        angle_ys.sort()
        self.angle_ys = [angle+np.pi/2 for angle in angle_ys]

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.randn((c.shape[0], 0))
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        # Perform volume rendering
        render_out = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last
        feature_samples, depth_samples, weights_samples = render_out['rgb'], render_out['depth'], render_out['weights']
        # feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}

    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)

    def set_fixed_condition(self, fixed_out_dict, rank):
        self.fixed_grid_z = torch.randn([fixed_out_dict['label'].shape[0], self.z_dim], device=f"cuda:{rank}")
        self.fixed_grid_c = torch.from_numpy(fixed_out_dict['label']).to(f"cuda:{rank}")
        self.fixed_real_images = fixed_out_dict['image']

    def set_random_condition(self, rnd_out_dict, rank):
        self.rnd_grid_z = torch.randn([rnd_out_dict['label'].shape[0], self.z_dim], device=f"cuda:{rank}")
        self.rnd_grid_c = torch.from_numpy(rnd_out_dict['label']).to(f"cuda:{rank}")
        self.rnd_real_images = rnd_out_dict['image']

    def get_input(self, out_dicts, training_sets, rank, batch_gpu, batch_size, phases):
        device = torch.device('cuda', rank)
        ## out_dicts['image'] : batch_gpu x 3 x H x W
        phase_real_img = (out_dicts['image'].to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)  # batch x 3 x H x W
        phase_real_c = out_dicts['label'].to(device).split(batch_gpu)  # batch x 25
        all_gen_z = torch.randn([len(phases) * batch_size, self.z_dim], device=device)
        all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
        all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for __ in range(len(phases)*(batch_size//batch_gpu)) for training_set in training_sets for _ in
                     range(batch_gpu//len(training_sets))]
        all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
        all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        real_inputs = {
            'real_img': list(phase_real_img),
            'real_c': list(phase_real_c)
        }
        gen_inputs = {
            'gen_z': list(all_gen_z),
            'gen_c': list(all_gen_c)
        }
        return real_inputs, gen_inputs, len(list(phase_real_img))

    def synthesis_images_for_display(self, global_step:int, print_angle=False, **kwargs):
        grid_z = (self.fixed_grid_z, self.rnd_grid_z)
        grid_c = (self.fixed_grid_c, self.rnd_grid_c)
        real_images = [self.fixed_real_images, self.rnd_real_images]
        grid_z = torch.cat([z for z in grid_z])  ##  batch x dim
        grid_c = torch.cat([c for c in grid_c])  ## batch x dim
        # pose_angles = compute_angle_from_matrix(grid_c)
        real_images = np.transpose(np.concatenate(real_images), (0, 2, 3, 1))  ## batch x dim x H x W

        trc_psh = 0.7
        trc_cutoff = 14
        intrinsics = grid_c[0:1, -9:].clone()
        angle_p = np.pi / 2
        angle_y_cnnk = np.pi / 2
        device = grid_z.get_device()
        cam_pivot = torch.tensor(self.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
        cam_radius = self.rendering_kwargs.get('avg_camera_radius', 2.7)

        if not self.rendering_kwargs['c_gen_conditioning_zero']:
            ws_grid_c = self.mapping(grid_z, grid_c, truncation_psi=trc_psh, truncation_cutoff=trc_cutoff)

        conditioning_cam2world_pose = LookAtPoseSampler.sample(angle_y_cnnk, angle_p, cam_pivot, radius=cam_radius, device=device)
        conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
        conditioning_params = conditioning_params.repeat(len(grid_z), 1)
        ws_cnnk = self.mapping(grid_z, conditioning_params, truncation_psi=trc_psh, truncation_cutoff=trc_cutoff)

        caption = f"[{global_step}] Row : real"
        Gvr_gridc = [[], [], []]

        if not self.rendering_kwargs['c_gen_conditioning_zero']:
            ## Gmn : grid_c, G_vr : grid_c
            out = self.synthesis(ws_grid_c, grid_c)
            Gvr_gridc[0].append((out['image'].cpu() * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy())  ## batch x dim x H x W
            Gvr_gridc[1].append((out['image_raw'].cpu() * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy())
            Gvr_gridc[2].append(-out['image_depth'].permute(0, 2, 3, 1).cpu().numpy())
            caption = f"{caption}, fake(real_pos)"

        ## Gmn : cnnk, G_vr : grid_c
        out = self.synthesis(ws_cnnk, grid_c)
        Gvr_gridc[0].append((out['image'].cpu() * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy())  ## batch x dim x H x W
        Gvr_gridc[1].append((out['image_raw'].cpu() * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy())
        Gvr_gridc[2].append(-out['image_depth'].permute(0, 2, 3, 1).cpu().numpy())
        caption = f"{caption}, fake(real_pos, cannonical_cond)"

        Gvr_rot = [[], [], []]

        if not self.rendering_kwargs['c_gen_conditioning_zero']:
            caption = f"{caption}, fake_zoomout(rotation)"
            ## Gmn : grid_c, G_vr : rot
            for angle_y in self.angle_ys:
                cam2world_pose = LookAtPoseSampler.sample(angle_y, angle_p, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
                camera_params = camera_params.repeat(len(grid_z), 1)
                img = self.synthesis(ws_grid_c, camera_params)

                # pose_angle = compute_angle_from_matrix(camera_params[0:1])
                # pose_angles.extend(pose_angle)
                image = (img['image'].permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
                raw = (img['image_raw'].permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
                depth = -img['image_depth'].permute(0, 2, 3, 1).cpu().numpy()
                Gvr_rot[0].append(image)
                Gvr_rot[1].append(raw)
                Gvr_rot[2].append(depth)

        ## Gmn : cnnk, G_vr : rot
        caption = f"{caption}, fake(rotation, cannonical_cond)"
        for angle_y in self.angle_ys:
            cam2world_pose = LookAtPoseSampler.sample(angle_y, angle_p, cam_pivot, radius=cam_radius, device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
            camera_params = camera_params.repeat(len(grid_z), 1)
            img = self.synthesis(ws_cnnk, camera_params)

            # pose_angle = compute_angle_from_matrix(camera_params[0:1])
            # pose_angles.extend(pose_angle)
            image = (img['image'].permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            raw = (img['image_raw'].permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            depth = -img['image_depth'].permute(0, 2, 3, 1).cpu().numpy()
            Gvr_rot[0].append(image)
            Gvr_rot[1].append(raw)
            Gvr_rot[2].append(depth)

        out_dict = {}
        for i, (rot_img, gridc_img, name) in enumerate(zip(Gvr_rot, Gvr_gridc, ['image', 'raw', 'depth'])):
            rot_img = np.concatenate(rot_img, 2)  ## batch x H x (W*angles) x ch
            gridc_img = np.concatenate(gridc_img, 2)  ## batch x H x (W*angles) x ch

            if i == 0:
                img = np.concatenate((real_images, gridc_img, rot_img), 2)
            else:
                img = np.concatenate((gridc_img, rot_img), 2)
            # if print_angle:
            #     for i, pa in enumerate(pose_angles):
            #         if i < 8:
            #             caption = f"{caption}\n 1 and 2 column in {i + 1} rows : {pa}"
            #         else:
            #             caption = f"{caption}\n {i - 8 + 3} column : {pa}"
            img = [out for out in img]
            img = np.concatenate(img, 0)
            out_dict[name] = {'image': img, 'caption': caption}
        # print(caption)
        return out_dict
