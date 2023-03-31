#!/bin/bash

# pretrained model for training
# Bash codes borrowd from https://github.com/KIMGEONUNG/BigColor
mkdir ckpts/ -pv

gdown "1CIwLBBnGRkZNV_5_Zs6sNFrqNWI37wf-" -O ckpts/hopenet_ckpt.pth
gdown "1iPB1lGSAT00Qq4h03G37TeKkGa9_6tfS" -O ckpts/EG3D_FFHQ_ckpt.pth