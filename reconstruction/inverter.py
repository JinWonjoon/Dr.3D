"""
    # Module for inverting real images to latent codes
    # Stage
    # 1) Project the input images to the latent space of the pre-trained model
    # 2) Joint-optimization of the pre-trained Generator, Pose-estimation network and the projected latent code
    # Codes are borrowed from Pivotal Tuning Inversion
    # https://github.com/danielroich/PTI
    
"""
import os
import torch
import copy
from tqdm import tqdm
import pdb
import PIL.Image

from src import dnnlib
from reconstruction.projectors import project
from reconstruction.locality_regularizer import Space_Regularizer

from lpips import LPIPS


class Inverter:
    def __init__(
        self, 
        data_loader,          # DataLoader of target images
        G,                    # Pre-trained Generator
        P,                    # Pre-trained Pose-estimation network
        device,               # cpu or cuda
        outdir,               # save directory..
        inverter_kwargs = {}, # Hyper-parameters for reconstruction
        proj_kwargs = {},     # Hyper-parameters for projection
        reg_kwargs = {},      # Hyper-parameters for regularization
        ):
        
        self.data_loader = data_loader
        self.G = G
        self.old_G = copy.deepcopy(G)
        self.P = P
        self.device = device
        self.outdir = outdir
        self.inverter_kwargs = dnnlib.EasyDict(inverter_kwargs)
        self.reg_kwargs = dnnlib.EasyDict(reg_kwargs)
        self.proj_kwargs = dnnlib.EasyDict(proj_kwargs)
        
        self.lpips_loss = LPIPS(net=self.inverter_kwargs.lpips_type).to(self.device).eval()
        self.space_regularizer = Space_Regularizer(self.G, self.lpips_loss, self.device, self.reg_kwargs)
        assert self.inverter_kwargs.max_images_to_invert >= len(self.data_loader), f"Use smaller number of images than {self.inverter_kwargs.max_images_to_invert}"

        self.l2_loss = torch.nn.MSELoss(reduction='mean')
        
    def get_projection(self, image, c, pose):
        id_image = torch.squeeze((image.to(self.device) + 1) / 2) * 255 # [-1,1] -> [0,255] for VGG loss
        ws = project(self.G, c, pose, id_image, device=self.device, w_avg_samples=600,
                                        num_steps=self.proj_kwargs.first_inv_steps,
                                        proj_kwargs=self.proj_kwargs)
        
        return ws
    
    
        
    
    def calc_loss(self, generated_images, real_images, ws_pivot, c_pivot):
        loss = 0.0

        if self.reg_kwargs.pt_l2_lambda > 0:
            l2_loss_val = self.l2_loss(generated_images, real_images)
            loss += l2_loss_val * self.reg_kwargs.pt_l2_lambda
        if self.reg_kwargs.pt_lpips_lambda > 0:
            loss_lpips = self.lpips_loss(generated_images, real_images)
            loss_lpips = torch.squeeze(loss_lpips)
            loss += loss_lpips * self.reg_kwargs.pt_lpips_lambda

        if self.inverter_kwargs.use_ball_holder and self.inverter_kwargs.use_locality_regularization:
            ball_holder_loss_val = self.space_regularizer.space_regularizer_loss(self.G, ws_pivot, c_pivot)
            loss += ball_holder_loss_val

        return loss, l2_loss_val, loss_lpips
    
    def invert(self, use_projected_w=False):
        
        pivots = {}
        images = []
        
        idx = 0
        # Project images to the latent space
        for out_dict in self.data_loader:
            image = out_dict['image']
            # image : numpy array, [0,255] uint8 -> torch.tensor, [-1,1]
            image = torch.tensor(image).to(self.device).to(torch.float32) / 127.5 - 1

            if use_projected_w:
                # LOAD.
                pass
            else:
                # Predict initial pose
                with torch.no_grad():
                    # pose_init = self.P(image).detach() # tensor. [yaw, pitch, roll]
                    real_pose, real_c = self.P.get_pose_label(image)
                    real_c = real_c.detach().to(self.device).to(torch.float32)
                
                ws_pivot = self.get_projection(image, real_c, real_pose)
            
            pivots[f'{idx}'] = (ws_pivot, real_c)
            images.append(image)
            
            save_path = os.path.join(self.outdir, f'proj_img_{idx:02d}.png')
            img_save = self.G.synthesis(ws_pivot, real_c)['image']
            img_save = (img_save[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img_save.cpu().numpy(), 'RGB').save(save_path)

            idx += 1
            
        
        # Tuning Generator following PTI
        self.optimizer = torch.optim.Adam(self.G.parameters(), lr=self.inverter_kwargs.pti_learning_rate)
        training_step = 0
        self.G.train()
        self.G.requires_grad_(True)
        
        for i in range(self.inverter_kwargs.max_pti_steps):
            generated_images_list = []
            idx = 0

            exit_flag1 = False
            exit_flag2 = False

            
            for image in images:
                ws_pivot, c_pivot = pivots[f'{idx}']

                real_images_batch = image.to(self.device)

                generated_images = self.G.synthesis(ws_pivot, c_pivot)['image']

                loss, l2_loss_val, loss_lpips = self.calc_loss(generated_images, real_images_batch, ws_pivot, c_pivot)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                use_ball_holder = training_step % self.inverter_kwargs.locality_regularization_interval == 0

                training_step += 1
                
                if i == self.inverter_kwargs.max_pti_steps - 1:
                    generated_images_list.append(generated_images)

                idx += 1
            
            print(f'PTI step {i + 1:>4d}/{self.inverter_kwargs.max_pti_steps}: loss {float(loss.item()):<5.2f}')
                            
            if exit_flag1 == True and exit_flag2 == True:
                break

        self.G.eval()
        
        with torch.no_grad():
            for idx, image in enumerate(images):
                ws_pivot, c_pivot = pivots[f'{idx}']
                save_path = os.path.join(self.outdir, 'images', f'final_img_{idx:02d}.png')
                os.makedirs(os.path.join(self.outdir, 'images'), exist_ok=True)
                img_save = self.G.synthesis(ws_pivot, c_pivot)['image']
                img_save = (img_save[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(img_save.cpu().numpy(), 'RGB').save(save_path)
        
        return self.G, pivots
            