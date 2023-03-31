import torch
import numpy as np
from src.camera_utils import sample_camera_positions
import pdb

class Space_Regularizer:
    def __init__(self, original_G, lpips_net, device, reg_kwargs={}):
        self.original_G = original_G
        self.morphing_regularizer_alpha = reg_kwargs.regularizer_alpha # hyperparameters.regulizer_alpha
        self.lpips_loss = lpips_net
        self.device = device
        self.reg_kwargs = reg_kwargs
        self.l2_loss = torch.nn.MSELoss(reduction='mean')

    def get_morphed_w_code(self, new_w_code, fixed_w):
        interpolation_direction = new_w_code - fixed_w
        interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
        direction_to_move = self.morphing_regularizer_alpha * interpolation_direction / interpolation_direction_norm
        result_w = fixed_w + direction_to_move
        self.morphing_regularizer_alpha * fixed_w + (1 - self.morphing_regularizer_alpha) * new_w_code

        return result_w

    def get_image_from_ws(self, w_codes, G):
        return torch.cat([G.synthesis(w_code, noise_mode='none', force_fp32=True) for w_code in w_codes])

    
    def ball_holder_loss_lazy(self, new_G, num_of_sampled_latents, ws_pivot, c_pivot):
        loss = 0.0
        # batch_size = self.reg_kwargs.reg_batch
        intrinsics = c_pivot[0:1, 16:].repeat([num_of_sampled_latents,1])
        extrinsics = sample_camera_positions(self.reg_kwargs.cam_pose_dicts, num_of_sampled_latents)
        new_c = torch.cat((extrinsics.to(intrinsics.get_device()), intrinsics), dim=-1)
        
        z_samples = np.random.randn(num_of_sampled_latents, self.original_G.z_dim)
        w_samples = self.original_G.mapping(torch.from_numpy(z_samples).to(self.device), new_c, truncation_psi=0.5)
        territory_indicator_ws = [self.get_morphed_w_code(w_code.unsqueeze(0), ws_pivot['ws_orig']) for w_code in w_samples['ws_orig']]
        territory_indicator_ws_deform = [self.get_morphed_w_code(w_code.unsqueeze(0), ws_pivot['ws_deform']) for w_code in w_samples['ws_deform']]
        new_c = list(new_c.split(1))

        for ws_orig, ws_deform, c in zip(territory_indicator_ws, territory_indicator_ws_deform, new_c):
            """
                # For the same latent codes and identical poses (which are randomly sampled),
                # old_G and new_G should generate the same images.
            """
            ws = {}
            ws['ws_orig'] = ws_orig
            ws['ws_deform'] = ws_deform
            new_img = new_G.synthesis(ws, c)['image']
            # synth_feat = new_G.synthesis(w_code[:,:new_G.synthesis.num_ws], noise_mode='none', force_fp32=True)
            # img_feature, img_raw, depth, cam2world_matrix, _, _, _ = new_G.implicit(synth_feat, pose, w_batch, z_cam)
            # new_img = new_G.superres(img_feature, w_code[:,new_G.synthesis.num_ws:], noise_mode='none', force_fp32=True)


            with torch.no_grad():
                old_img = self.original_G.synthesis(ws, c)['image']
                # old_synth_feat = self.original_G.synthesis(w_code[:,:self.original_G.synthesis.num_ws], noise_mode='none', force_fp32=True)
                # old_img_feature, _, _, _, _, _, _ = self.original_G.implicit(old_synth_feat, pose, w_batch, z_cam)
                # old_img = self.original_G.superres(old_img_feature, w_code[:,self.original_G.synthesis.num_ws:], noise_mode='none', force_fp32=True)
                

            if self.reg_kwargs.regularizer_l2_lambda > 0:
                l2_loss_val = self.l2_loss(old_img, new_img)
                loss += l2_loss_val * self.reg_kwargs.regularizer_l2_lambda

            if self.reg_kwargs.regularizer_lpips_lambda > 0:
                loss_lpips = self.lpips_loss(old_img, new_img)
                loss_lpips = torch.mean(torch.squeeze(loss_lpips))
                loss += loss_lpips * self.reg_kwargs.regularizer_lpips_lambda

        return loss / len(territory_indicator_ws)

    def space_regularizer_loss(self, new_G, ws_pivot, c_pivot):
        ret_val = self.ball_holder_loss_lazy(new_G, self.reg_kwargs.latent_ball_num_of_samples, ws_pivot, c_pivot)
        return ret_val