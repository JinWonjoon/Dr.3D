"""
    # Network Architecture for Deformation-aware Network. (Sec.4.1 in the main paper of Dr.3D)
    # IN: deformation code z_d
    # Condition: deformation level (resolution in configuration, e.g. [4,8,16,32])
    # OUT: 2D feature maps to modulate generator features for each level
"""
import pdb

import numpy as np
import torch
from src.torch_utils import persistence
from src.torch_utils import misc
from src.training.networks.networks_stylegan2 import FullyConnectedLayer
from src import dnnlib

@persistence.persistent_class
class DeformNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        c_dim,                      # Input Condition (C) dimensionality.
        deform_res,                 # Output resolution. Represented by feature resolution. List.
        mapping_kwargs       = {},  # Arguments for MappingNetwork.
        additional_kwargs    = {},  # Additional Kwargs.
    ):
        super().__init__()
        
        self.additional_kwargs = additional_kwargs
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.c_dim = c_dim
        for res in deform_res:
            assert res%2 == 0 and res>=4 and res<=32, f"Not adqueate deform_res {deform_res}. It should be included in the set of [4,8,16,32]"
        self.num_ws = len(deform_res) * 2
        self.deform_res = deform_res # List. e.g.) [8,16,32].
        
        for res in self.deform_res:
            in_channels = self.w_dim
            out_channels = res * res + 512 # 512 means channel dimension of the generator's feature maps
            block0 = FullyConnectedLayer(in_channels, out_channels, bias_init=0)
            block1 = FullyConnectedLayer(in_channels, out_channels, bias_init=0)
            # Weight initialization.
            block0.weight.data = block0.weight.data * 0.01
            block1.weight.data = block1.weight.data * 0.01
            block_list = torch.nn.ModuleList([block0, block1])
            
            setattr(self, f'b{res}', block_list)
        
        mapping_kwargs.update({'num_ws': self.num_ws, 'c_dim': c_dim})
        self.mapping = dnnlib.util.construct_class_by_name(**mapping_kwargs)
        
    def get_expanded_kernel(self, residual, res):
        a = residual[...,:res*res].reshape(-1,1,res,res)
        b = residual[...,res*res:].reshape(-1,512,1,1)

        # [N, 512, length, length]
        return a * b

    def get_deform_feature(self, ws, **synthesis_kwargs):
        misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
        ws = ws.to(torch.float32) # [batch, len(deform_res)*2, 256]
        feature_maps = {}
        for idx, res in enumerate(self.deform_res):
            ws_res = ws.narrow(1, 2*idx, 2)
            ws_iter = iter(ws_res.unbind(dim=1))
            block_list = getattr(self, f'b{res}')
            residual_0 = self.get_expanded_kernel(block_list[0](next(ws_iter)), res)
            residual_1 = self.get_expanded_kernel(block_list[1](next(ws_iter)), res)
            feature_maps[f'm{res}_conv0'] = residual_0
            feature_maps[f'm{res}_conv1'] = residual_1
        
        return feature_maps

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        feature_maps = self.get_deform_feature(ws, **synthesis_kwargs)
        return feature_maps

#----------------------------------------------------------------------------