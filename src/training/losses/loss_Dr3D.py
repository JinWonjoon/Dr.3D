# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
import pdb
import PIL.Image

from src.torch_utils import training_stats
from src.torch_utils.ops import conv2d_gradfix
from src.torch_utils.ops import upfirdn2d
from src.training.discriminators.dual_discriminator import filtered_resizing
from src.training.losses.geometric_priors import depthSimilarityLoss, normalSmoothLoss

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss_Dr3D(Loss):
    def __init__(self, device, G, D, G_freeze, P, P_freeze, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0,
                 pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0,
                 r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64,
                 neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0,
                 gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased',
                 additional_kwargs={}):

        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.G_freeze           = G_freeze
        self.P                  = P
        self.P_freeze           = P_freeze
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        self.additional_kwargs = additional_kwargs
        self.depthSimilarityLoss = depthSimilarityLoss(k_std=additional_kwargs['k_std'], near=additional_kwargs['near'], far=additional_kwargs['far'], mode=additional_kwargs['depth_mode'])
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        return gen_output, ws

    def run_G_freeze(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G_freeze.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G_freeze.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G_freeze.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        return gen_output['image_depth']

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False, save_augment_img_path=None):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            if save_augment_img_path is not None:
                save_augment_img_name = save_augment_img_path
                img_save = torch.cat((torch.cat(img['image'].split(1), dim=-1), torch.cat(augmented_pair[:, :img['image'].shape[1]].split(1), dim=-1)), dim=-2)
                img_save = (img_save[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(img_save.cpu().numpy(), 'RGB').save(save_augment_img_name)
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)


        # use_aug = [1 for aug_pipe in self.augment_pipes if aug_pipe is not None]
        # if len(use_aug) > 0:   ## dict!
        #     num_domains = len(self.augment_pipes)
        #     imgs = []
        #     img_raws = []
        #     cs = []
        #     ds = []
        #     for domain in range(num_domains):
        #         domain_idx = torch.nonzero(d)  # batch_idx, domain_idx
        #         batch_idxs = (domain_idx[:, 1] == domain).nonzero(as_tuple=True)[0]
        #         if self.augment_pipes[domain] is not None and len(batch_idxs) > 0:
        #             img_ = img['image'][batch_idxs]
        #             img_raw_ = torch.nn.functional.interpolate(img['image_raw'][batch_idxs], size=img_.shape[2:], mode='bilinear', antialias=True)
        #             augmented_pair = self.augment_pipes[domain](torch.cat([img_, img_raw_], dim=1))
        #             imgs.append(augmented_pair[:, :img_.shape[1]])
        #             img_raws.append(torch.nn.functional.interpolate(augmented_pair[:, img_.shape[1]:], size=img['image_raw'][batch_idxs].shape[2:], mode='bilinear', antialias=True))
        #             cs.append(c[batch_idxs])
        #             ds.append(d[batch_idxs])
        #             if save_augment_img_path is not None:
        #                 save_augment_img_name = save_augment_img_path.replace(".png", f'_domain{domain}.png')
        #                 img_save = torch.cat((torch.cat(img['image'][batch_idxs].split(1), dim=-1), torch.cat(augmented_pair[:, :img_.shape[1]].split(1), dim=-1)), dim=-2)
        #                 img_save = (img_save[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        #                 PIL.Image.fromarray(img_save.cpu().numpy(), 'RGB').save(save_augment_img_name)
        #         else:
        #             imgs.append(img['image'][batch_idxs])
        #             img_raws.append(img['image_raw'][batch_idxs])
        #             cs.append(c[batch_idxs])
        #             ds.append(d[batch_idxs])
        #             # img['image'][split_ * domain:split_ * (domain + 1)] = augmented_pair[:, :img_.shape[1]]
        #             # img['image_raw'][split_ * domain:split_ * (domain + 1)] = torch.nn.functional.interpolate(augmented_pair[:, img_.shape[1]:], size=img['image_raw'][split_ * domain:split_ * (domain + 1)].shape[2:], mode='bilinear', antialias=True)
        #     img['image'] = torch.cat(imgs, 0)
        #     img['image_raw'] = torch.cat(img_raws, 0)
        #     c = torch.cat(cs, 0)
        #     d = torch.cat(ds, 0)

        logits = self.D(img, c, update_emas=update_emas)
        return logits

    #def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
    def accumulate_gradients(self, phase, inputs, gain, cur_nimg, save_augment_img_path=None):
        real_img = inputs['real_img']
        real_c = inputs['real_c']
        real_pose = inputs['real_pose']
        gen_z = inputs['gen_z']
        gen_c = inputs['gen_c']

        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth', 'Pboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                training_stats.report(f'Loss_{phase}/scores/fake', gen_logits)
                training_stats.report(f'Loss_{phase}/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report(f'Loss_{phase}/G/loss', loss_Gmain)
            
            """
                Geometric Priors. Sec.4.3 in the paper.
            """
            # Depth similarity loss
            depth_reg = 0
            if self.additional_kwargs.get('depth_reg', 0) > 0:
                with torch.no_grad():
                    prior_depth = self.run_G_freeze(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                depth_reg_ = self.depthSimilarityLoss(gen_img['image_depth'], prior_depth, prior_depth.device) # Gaussian + MSE
                depth_reg = depth_reg_ * self.additional_kwargs['depth_reg']
                training_stats.report(f'Loss_{phase}/G/depth_smooth_loss', depth_reg)
                
            # Pose loss
            pose_reg = 0
            if self.additional_kwargs.get('pose_reg', 0) > 0:
                pred_pose = self.P(gen_img['image'])
                pose_reg_ = torch.nn.MSELoss()(pred_pose, real_pose)
                pose_reg = pose_reg_ * self.additional_kwargs['pose_reg']
                training_stats.report(f'Loss_{phase}/G/pose_loss', pose_reg)
                
            # Normal smoothness loss
            normal_reg = 0
            if self.additional_kwargs.get('normal_reg', 0) > 0:
                normal_reg_ = normalSmoothLoss(gen_img['image_depth'])
                normal_reg = normal_reg_ * self.additional_kwargs['normal_reg']
                training_stats.report(f'Loss_{phase}/G/normal_smoothness_loss', normal_reg)
                
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain.mean() + depth_reg + pose_reg + normal_reg).mul(gain).backward()


        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] in ['l1', 'l2']:
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0: # Not used..
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(gen_z), gen_c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((gen_z.shape[0], 1000, 3), device=gen_z.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            if self.G.rendering_kwargs['reg_type'] == 'l1':
                TVloss_ = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed)
            elif self.G.rendering_kwargs['reg_type'] == 'l2':
                TVloss_ = torch.nn.functional.mse_loss(sigma_initial, sigma_perturbed)
            else:
                NotImplementedError
            TVloss = TVloss_ * self.G.rendering_kwargs['density_reg']
            training_stats.report(f'Loss_{phase}/TVloss', TVloss_)
            training_stats.report(f'Loss_{phase}/G/TVloss_den', TVloss)
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((gen_z.shape[0], 2000, 3), device=gen_z.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=gen_z.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(gen_z), gen_c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((gen_z.shape[0], 1000, 3), device=gen_z.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)

            initial_coordinates = torch.rand((gen_z.shape[0], 2000, 3), device=gen_z.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=gen_z.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(gen_z), gen_c, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((gen_z.shape[0], 1000, 3), device=gen_z.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report(f'Loss_{phase}/scores/fake', gen_logits)
                training_stats.report(f'Loss_{phase}/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma, save_augment_img_path=save_augment_img_path)
                training_stats.report(f'Loss_{phase}/scores/real', real_logits)
                training_stats.report(f'Loss_{phase}/signs/real', real_logits.sign())
                if self.augment_pipe is not None:
                    training_stats.report(f'Loss_aug/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report(f'Loss_{phase}/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report(f'Loss_{phase}/r1_penalty', r1_penalty)
                    training_stats.report(f'Loss_{phase}/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()
    
        """
            Alternating adaptation of the pose-estimation networks. Sec.4.2 in the paper.
        """
        if phase in ['Pboth']:
            with torch.no_grad():
                # sample random pose True  -> pose_sampled comes from gaussian
                # sample random pose False -> pose_sampled comes from P_freeze 
                pseudo_pose, pseudo_label = self.P_freeze.get_pose_label(real_img['image']) # batch x 25
                pseudo_pose = pseudo_pose.detach().to(real_img['image'].device).to(torch.float32)
                pseudo_label = pseudo_label.detach().to(real_img['image'].device).to(torch.float32)
                pseudo_gen_img, _ = self.run_G(gen_z, pseudo_label, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                
            pred_pose_for_adaptation = self.P(pseudo_gen_img['image']) # detach computational graph. [b,2]
            loss_Pmain = torch.nn.MSELoss()(pred_pose_for_adaptation, pseudo_pose)
            training_stats.report('Loss/P/loss', loss_Pmain)

#----------------------------------------------------------------------------
