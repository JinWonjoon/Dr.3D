# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
Helper functions for constructing camera parameter matrices. Primarily used in visualization and inference scripts.
"""

import math
from scipy.spatial.transform import Rotation as R
import pdb

import torch
import torch.nn as nn
import numpy as np
import copy

from src.training.volumetric_rendering import math_utils

class GaussianCameraPoseSampler:
    """
    Samples pitch and yaw from a Gaussian distribution and returns a camera pose.
    Camera is specified as looking at the origin.
    If horizontal and vertical stddev (specified in radians) are zero, gives a
    deterministic camera pose with yaw=horizontal_mean, pitch=vertical_mean.
    The coordinate system is specified with y-up, z-forward, x-left.
    Horizontal mean is the azimuthal angle (rotation around y axis) in radians,
    vertical mean is the polar angle (angle from the y axis) in radians.
    A point along the z-axis has azimuthal_angle=0, polar_angle=pi/2.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = GaussianCameraPoseSampler.sample(math.pi/2, math.pi/2, radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)


class LookAtPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    camera is specified as looking at 'lookat_position', a 3-vector.

    Example:
    For a camera pose looking at the origin with the camera at position [0, 0, 1]:
    cam2world = LookAtPoseSampler.sample(math.pi/2, math.pi/2, torch.tensor([0, 0, 0]), radius=1)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, lookat_position, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu', h=None, v=None):
        if h is None:
            h = torch.randn((batch_size, 1), device=device) * horizontal_stddev + horizontal_mean
        if v is None:
            v = torch.randn((batch_size, 1), device=device) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        # forward_vectors = math_utils.normalize_vecs(-camera_origins)
        forward_vectors = math_utils.normalize_vecs(lookat_position - camera_origins)
        c2w = create_cam2world_matrix(forward_vectors, camera_origins)
        return c2w

class UniformCameraPoseSampler:
    """
    Same as GaussianCameraPoseSampler, except the
    pose is sampled from a uniform distribution with range +-[horizontal/vertical]_stddev.

    Example:
    For a batch of random camera poses looking at the origin with yaw sampled from [-pi/2, +pi/2] radians:

    cam2worlds = UniformCameraPoseSampler.sample(math.pi/2, math.pi/2, horizontal_stddev=math.pi/2, radius=1, batch_size=16)
    """

    @staticmethod
    def sample(horizontal_mean, vertical_mean, horizontal_stddev=0, vertical_stddev=0, radius=1, batch_size=1, device='cpu'):
        h = (torch.rand((batch_size, 1), device=device) * 2 - 1) * horizontal_stddev + horizontal_mean
        v = (torch.rand((batch_size, 1), device=device) * 2 - 1) * vertical_stddev + vertical_mean
        v = torch.clamp(v, 1e-5, math.pi - 1e-5)

        theta = h
        v = v / math.pi
        phi = torch.arccos(1 - 2*v)

        camera_origins = torch.zeros((batch_size, 3), device=device)

        camera_origins[:, 0:1] = radius*torch.sin(phi) * torch.cos(math.pi-theta)
        camera_origins[:, 2:3] = radius*torch.sin(phi) * torch.sin(math.pi-theta)
        camera_origins[:, 1:2] = radius*torch.cos(phi)

        forward_vectors = math_utils.normalize_vecs(-camera_origins)
        return create_cam2world_matrix(forward_vectors, camera_origins)    

def create_cam2world_matrix(forward_vector, origin):
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    Works on batches of forward_vectors, origins. Assumes y-axis is up and that there is no camera roll.
    """

    forward_vector = math_utils.normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=origin.device).expand_as(forward_vector)

    right_vector = -math_utils.normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = math_utils.normalize_vecs(torch.cross(forward_vector, right_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=origin.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :, :]
    assert(cam2world.shape[1:] == (4, 4))
    return cam2world


def FOV_to_intrinsics(fov_degrees, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """

    focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics

def compute_angle_from_matrix(Ms):
    pose_angles = []
    for M in Ms:
        M = torch.reshape(M[:16], (4, 4)).data.cpu().numpy()
        r = R.from_matrix(M[:3, :3])
        ypr_deg = r.as_euler('zyx', degrees=True)
        pose = M[:3, 3].tolist()
        pose_angle = ", ".join([f"{i:0.3f}" for i in pose])
        angles = ", ".join([f"{deg:0.1f}" for deg in ypr_deg])
        pose_angle = f"pose: ({pose_angle}), angle: ({angles})"
        pose_angles.append(pose_angle)
    return pose_angles

### by karmina from pigan
def sample_camera_positions(cam_pose_dicts, n=1, r=2.7, lookat_position=[0,0,0]):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    """
    keys = ['pitch', 'yaw', 'roll']
    rads = torch.zeros((n, 3))

    for i, key in enumerate(keys):
        value = cam_pose_dicts[key]
        mode = value['mode']
        std = value['std']
        mean = value['mean']
        if mode == 'normal' or mode == 'gaussian':
            rad = torch.randn((n, 1)) * std + mean

        elif mode == 'spherical_uniform':
            if key == 'yaw':
                if isinstance(std, list):
                    rad = (torch.rand((n, 1)) - .5) * 2 * (std[1] - std[0]) #+ mean  # -0.599+1.556  (0.957) ~ 0.599 + 1.556 (2.155)
                    positive_idxs = (rad[:, 0] >= 0).nonzero(as_tuple=True)[0]
                    negative_idxs = (rad[:, 0] < 0).nonzero(as_tuple=True)[0]
                    rad[positive_idxs] = rad[positive_idxs] + std[0]
                    rad[negative_idxs] = rad[negative_idxs] - std[0]
                    rad = rad + mean
                else:
                    rad = (torch.rand((n, 1)) - .5) * 2 * std + mean  # -0.599+1.556  (0.957) ~ 0.599 + 1.556 (2.155)
            else:
                if key == 'pitch':
                    std, mean = std / math.pi, mean / math.pi

                rad = (torch.rand((n, 1)) - .5) * 2 * std + mean        # -0.599+1.556  (0.957) ~ 0.599 + 1.556 (2.155)

                if key == 'pitch':
                    rad = torch.clamp(rad, 1e-5, 1 - 1e-5)
                    rad = torch.arccos(1 - 2 * rad)
        else:
            # Just use the mean.
            rad = torch.ones((n, 1), dtype=torch.float) * mean

        if key == 'pitch':
            rad = torch.clamp(rad, 1e-5, math.pi - 1e-5)
        # elif key == 'yaw':
        #     rad = torch.clamp(rad, 0, math.pi * 2)

        rads[:, i:i+1] = rad

    # lookat_position = torch.zeros((n, 3))
    lookat_position = torch.tensor([lookat_position]).repeat(n, 1)
    camera_origins = torch.zeros((n, 3))

    ## 0 : pitch, 1: yaw, 2: roll
    camera_origins[:, 0] = r * torch.sin(rads[:, 0]) * torch.cos(math.pi - rads[:, 1])
    camera_origins[:, 2] = r * torch.sin(rads[:, 0]) * torch.sin(math.pi - rads[:, 1])
    camera_origins[:, 1] = r * torch.cos(rads[:, 0])

    forward_vectors = math_utils.normalize_vecs(lookat_position - camera_origins)
    extrinsics = create_cam2world_matrix(forward_vectors.cuda(), camera_origins.cuda()).data.cpu().numpy()

    for i, (extrinsic, roll) in enumerate(zip(extrinsics, rads[:, 2])):
        r = R.from_matrix(extrinsic[:3, :3])
        rot_rad = r.as_euler('zyx')
        new_rot_rad = R.from_euler('zyx', np.array([rot_rad[0] - roll, rot_rad[1], rot_rad[2]]))
        new_rot_mat = new_rot_rad.as_matrix()
        extrinsics[i, :3, :3] = new_rot_mat

    extrinsics = torch.from_numpy(extrinsics.reshape(len(extrinsics),-1))
    return extrinsics


## for random camera pose
def sampling_random_cam(c, cam_pose_dicts):
    intrinsics = c[:, 16:]
    extrinsics = sample_camera_positions(cam_pose_dicts, len(c))
    new_c = torch.cat((extrinsics.to(intrinsics.get_device()), intrinsics), dim=-1)
    return new_c

def sampling_random_cam_specific_pose(c, cam_pose_dicts, pose_rad:dict):
    intrinsics = c[:, 16:]
    cam_pose_dicts = copy.deepcopy(cam_pose_dicts)
    for angle_name, angle_value in pose_rad.items():
        cam_pose_dicts[angle_name]['mean'] = angle_value
        cam_pose_dicts[angle_name]['std'] = 0

    extrinsics = sample_camera_positions(cam_pose_dicts, len(c))
    new_c = torch.cat((extrinsics.to(intrinsics.get_device()), intrinsics), dim=-1)
    return new_c

#------------------------------------------------------------------------------------

def fix_intrinsics():
    intrinsics = np.eye(3)
    intrinsics[0,0] = 2985.29/700
    intrinsics[1,1] = 2985.29/700
    intrinsics[0,2] = 1/2
    intrinsics[1,2] = 1/2
    assert intrinsics[0,1] == 0
    assert intrinsics[2,2] == 1
    assert intrinsics[1,0] == 0
    assert intrinsics[2,0] == 0
    assert intrinsics[2,1] == 0
    return intrinsics

def normalize_vecs(vectors: np.array) -> np.array:
    """
    Normalize vector lengths.
    """
    return vectors / np.linalg.norm(vectors)

def create_cam2world_matrix_np(forward_vector, origin):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""
    forward_vector = normalize_vecs(forward_vector)
    up_vector = np.array([0., 1., 0.])

    right_vector = -normalize_vecs(np.cross(up_vector, forward_vector, axis=-1))
    up_vector = normalize_vecs(np.cross(forward_vector, right_vector, axis=-1))

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = np.stack((right_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = origin
    cam2world = (translation_matrix @ rotation_matrix)[:, :]
    assert (cam2world.shape == (4, 4))
    return cam2world

def sample(yaw, pitch):
    radius = 2.7
    lookat_position = np.zeros(3)

    theta = yaw
    pitch = pitch / math.pi  # pitch
    phi = np.arccos(1 - 2 * pitch)

    camera_origins = np.zeros(3)

    camera_origins[0:1] = radius * np.sin(phi) * np.cos(math.pi - theta)
    camera_origins[2:3] = radius * np.sin(phi) * np.sin(math.pi - theta)
    camera_origins[1:2] = radius * np.cos(phi)

    forward_vectors = normalize_vecs(lookat_position - camera_origins)
    return create_cam2world_matrix_np(forward_vectors, camera_origins)

