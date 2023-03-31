# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Miscellaneous utilities used internally by the quality metrics."""

import os
import pdb
import time
import math
import hashlib
import pickle
import copy
import uuid
import numpy as np
import PIL.Image
import torch
import sys
sys.path.append("./src")
from src import dnnlib
from src.camera_utils import sample_camera_positions

#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, P=None, G_kwargs={}, dataset_kwargs={}, num_gpus=1, rank=0, device=None, progress=None, cache=True, save_genimg_name=None, Gvr_yaw_std=-1, Gmn_cam_dist_info=None, gvr_yaw_range=None):
        assert 0 <= rank < num_gpus
        self.G = G
        self.P = P
        self.G_kwargs = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus = num_gpus
        self.rank = rank
        self.device = device if device is not None else torch.device('cuda', rank)
        self.progress = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache = cache
        self.save_genimg_name = save_genimg_name
        self.Gvr_yaw_std = Gvr_yaw_std    # default -1
        self.Gmn_cam_dist_info = Gmn_cam_dist_info  # default None
        self.gvr_yaw_range = gvr_yaw_range

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = pickle.load(f).to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

def iterate_random_images(P, opts, batch_size):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    while True:
        images = [dataset.get_image(np.random.randint(len(dataset))) for _i in range(batch_size)]
        images = torch.from_numpy(np.stack(images)).pin_memory().to(opts.device)
        images = images / 127.5 - 1
        with torch.no_grad():
            _, c = P.get_pose_label(images)
            c = c.detach().to(opts.device).to(torch.float32)
        yield c

def iterate_random_labels_cond(opts, batch_size):
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        while True:
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            yield c

def iterate_random_priors_cond(opts, batch_size, cam_dist_info, lookat_position):
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        while True:
            c = dataset.get_label(np.random.randint(1))
            intrinsics = c[16:]
            extrinsics = sample_camera_positions(cam_dist_info, batch_size, lookat_position=lookat_position)
            intrinsics = intrinsics.unsqueeze(0).repeat(len(extrinsics), 1)
            c = torch.cat((extrinsics.to(opts.device), intrinsics.to(opts.device)), dim=-1)
            yield c


#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for out_dicts in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        images = out_dicts["image"]
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and labels.
    c_iter_data, c_iter_prior_mn, c_iter_prior_vr = None, None, None
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    P = copy.deepcopy(opts.P).eval().requires_grad_(False).to(opts.device)
    # if opts.Gvr_yaw_std == -1 or opts.Gmn_cam_dist_info is None: ## sampling from data
    # c_iter_data = iterate_random_labels_cond(opts=opts, batch_size=batch_gen)
    c_iter_est = iterate_random_images(P, opts=opts, batch_size=batch_gen)

    if opts.gvr_yaw_range is not None:
        lookat_position = G.rendering_kwargs['avg_camera_pivot']
        cam_dist_info = {'pitch': {'mode': 'gaussian', 'mean': np.pi / 2, 'std': 0.260},
                         'yaw': {'mode': 'spherical_uniform', 'mean': np.pi / 2, 'std': [g / 180 * np.pi for g in opts.gvr_yaw_range]},
                         'roll': {'mode': 'gaussian', 'mean': 0, 'std': 0.054}, }
        c_iter_prior_vr = iterate_random_priors_cond(opts=opts, batch_size=batch_gen, cam_dist_info=cam_dist_info, lookat_position=lookat_position)
    elif opts.Gvr_yaw_std > 0:   ## sampling from prior for Gvr
        lookat_position = G.rendering_kwargs['avg_camera_pivot']
        cam_dist_info = {'pitch':{'mode': 'gaussian', 'mean': np.pi/2, 'std': 0.260},
                         'yaw': {'mode': 'spherical_uniform', 'mean': np.pi / 2, 'std': opts.Gvr_yaw_std/180 * np.pi},
                         'roll': {'mode': 'gaussian', 'mean': 0, 'std': 0.054},}
        c_iter_prior_vr = iterate_random_priors_cond(opts=opts, batch_size=batch_gen, cam_dist_info=cam_dist_info, lookat_position=lookat_position)
    # if opts.Gmn_cam_dist_info is not None:   ## sampling from prior
    #     c_iter_prior_mn = iterate_random_priors_cond(opts=opts, batch_size=batch_gen, cam_dist_info=opts.Gmn_cam_dist_info)


    # Initialize.
    stats = FeatureStats(**stats_kwargs)
    assert stats.max_items is not None
    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    G_kwargs_mapping = copy.deepcopy(opts.G_kwargs)
    G_kwargs_mapping.update({
        'truncation_psi': 1,
        'truncation_cutoff': None,
        'update_emas': False,
    })
    G_kwargs_synthesis = copy.deepcopy(opts.G_kwargs)
    G_kwargs_synthesis.update({
        'neural_rendering_resolution': None,
        'update_emas': False,
        'cache_backbone': False,
        'use_cached_backbone': False
    })

    while not stats.is_full():
        images = []
        for _i in range(batch_size // batch_gen):

            if opts.Gvr_yaw_std == -1 or opts.Gmn_cam_dist_info is None: ## sampling from pose estimation networks
                c_gmn = next(c_iter_est)
            # c_gmn = c_est #if opts.Gmn_cam_dist_info is None else next(c_iter_prior_mn)
            if c_iter_prior_vr is None:
                c_gvr = c_gmn
            else:
                c_gvr = next(c_iter_prior_vr)

            inputs_mn = {'z': torch.randn([batch_gen, G.z_dim], device=opts.device), 'c': c_gmn}
            inputs_syn = {'c': c_gvr}
            ws = G.mapping(**inputs_mn, **G_kwargs_mapping)
            inputs_syn['ws'] = ws
            img = G.synthesis(**inputs_syn, **G_kwargs_synthesis)['image']
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            images.append(img)

        images = torch.cat(images)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        if opts.save_genimg_name is not None:
            num_imgs = min(images.shape[0], 64)
            n_img = int(math.sqrt(num_imgs))
            img_ = images[:n_img * n_img].clone().permute(0, 2, 3, 1)
            save_img = None
            for kk in range(n_img):
                save_img_ = torch.cat([img_[n_img * kk + jj] for jj in range(n_img)], dim=1)
                save_img = torch.cat((save_img, save_img_), dim=0) if save_img is not None else save_img_
            PIL.Image.fromarray(save_img.cpu().numpy(), 'RGB').save(opts.save_genimg_name)

        features = detector(images, **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)
    return stats

#----------------------------------------------------------------------------
