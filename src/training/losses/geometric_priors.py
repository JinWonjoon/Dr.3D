import torch
import torch.nn as nn
import numpy as np
from scipy import signal
import torch.nn.functional as f
# GAN output : [N, 1, H, W]
# MiDAS input : [N, H, W]

class depthSimilarityLoss(nn.Module):
    def __init__(self, k_std = 3, near = 2.25, far = 3.3, mode='MSE'):
        super().__init__()

        # window size = 3 X std
        # k_std should be odd number!!
        assert k_std % 2 != 0

        k_size = 3 * k_std
        kernel_1d = signal.gaussian(k_size, std = k_std)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = np.outer(kernel_1d, kernel_1d.transpose())

        self.pad = k_size // 2
        self.weight = torch.tensor(kernel_2d, dtype=torch.float32).view(1,1,k_size,k_size).contiguous().requires_grad_(False) # [out_ch=1, in_ch=1, k_size, k_size]
    
        self.near = near
        self.far = far
        self.mode = mode
        
        assert mode in ['MSE', 'HUBER', 'L1'], f'depthSimilarityLoss has not supported mode : {mode}' 

    def forward(self, depth, depth_GT, device):
        # input : [N, 1, H, W]
        # conv2d -> squeeze(1) -> MiDAS_loss

        weight = self.weight.to(device)
        depth_pad = f.pad(depth, pad=(self.pad,self.pad, self.pad, self.pad), mode='replicate')
        depth_blur = f.conv2d(depth_pad, weight, bias=None, stride=1)
        depth_GT_pad = f.pad(depth_GT, pad=(self.pad,self.pad, self.pad, self.pad), mode='replicate')
        depth_GT_blur = f.conv2d(depth_GT_pad, weight, bias=None, stride=1)

        mask1 = self.near < depth
        mask1[depth > self.far] = False
        mask2 = self.near < depth_GT
        mask2[depth_GT > self.far] = False

        mask = mask1 & mask2

        if self.mode == 'MSE':
            loss = torch.nn.MSELoss(reduction='none')(depth_blur, depth_GT_blur)
            loss = (loss * mask.float()).sum()
            return loss / mask.sum()
        
        elif self.mode == 'L1':
            loss = torch.nn.L1Loss(reduction='none')(depth_blur, depth_GT_blur)
            loss = (loss * mask.float()).sum()
            return loss / mask.sum()
        
        elif self.mode == 'HUBER':
            loss = torch.nn.SmoothL1Loss(reduction='none')(depth_blur, depth_GT_blur)
            loss = (loss * mask.float()).sum()
            return loss / mask.sum()

def normalSmoothLoss(depth, scale=100):
    # depth shape : [N, 1, H, W]
    # we need [N, H, W]

    # scaling depth
    depth = depth.squeeze(1) * scale 

    # gradient of depth
    grad_depth_y, grad_depth_x = torch.gradient(depth, dim=(1,2))
    normal = torch.stack([-grad_depth_x, -grad_depth_y, torch.ones_like(grad_depth_x)], axis=1)
    n = torch.norm(normal, dim=1)

    # Normal shape : [N, 3, H, W]
    normal = normal / n.unsqueeze(1)
    

    # gradient of normal
    grad_normal_y, grad_normal_x = torch.gradient(normal, dim=(2,3))

    grad_normal = torch.abs(grad_normal_x).mean() + torch.abs(grad_normal_y).mean()

    return grad_normal