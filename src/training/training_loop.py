# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import pdb
from src import dnnlib

from src.torch_utils import misc
from src.torch_utils import training_stats
from src.torch_utils.ops import conv2d_gradfix
from src.torch_utils.ops import grid_sample_gradfix
from src import legacy
from src.training.pose_networks.hopenet import Posenet, get_ignored_params, get_non_ignored_params, get_fc_params


#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_sets:list, all_indices:list, fix_indices:list, random_seed=0, gw=None, gh=None):
    """
    training_sets : [{}, {}, {}]
    """
    rnd = np.random.RandomState(random_seed)
    grid_indices = []
    out_dicts = {}
    # No labels => show random subset of training samples.

    for jj, (all_indice, fix_indice, training_set) in enumerate(zip(all_indices, fix_indices, training_sets)):
        if len(all_indice) < (gw*gh):
            all_indice = [i for i in range(len(training_set)) if i not in fix_indice]
            rnd.shuffle(all_indice)
            all_indices[jj] = all_indice
        grid_indices.append(all_indice[:gw*gh])
        # Load data.
        for i in all_indice[:gw*gh]:
            out_dict = training_set[i]  ## key : ['image', 'label', 'domain']
            for k, v in out_dict.items():
                if k in out_dicts.keys():
                    out_dicts[k].append(v)
                else:
                    out_dicts[k] = [v]
    all_indices = [all_indice[(gw * gh):] for all_indice in all_indices]
    for k,v in out_dicts.items():
        out_dicts[k] = np.stack(v)
    
    return out_dicts, all_indices, grid_indices

#----------------------------------------------------------------------------
def setup_snapshot_image_grid_Dr3D(training_sets:list, all_indices:list, fix_indices:list, P, rank, random_seed=0, gw=None, gh=None):
    """
    training_sets : [{}, {}, {}]
    """
    device = torch.device('cuda', rank)
    
    rnd = np.random.RandomState(random_seed)
    grid_indices = []
    out_dicts = {}
    # No labels => show random subset of training samples.

    for jj, (all_indice, fix_indice, training_set) in enumerate(zip(all_indices, fix_indices, training_sets)):
        if len(all_indice) < (gw*gh):
            all_indice = [i for i in range(len(training_set)) if i not in fix_indice]
            rnd.shuffle(all_indice)
            all_indices[jj] = all_indice
        grid_indices.append(all_indice[:gw*gh])
        # Load data.
        for i in all_indice[:gw*gh]:
            out_dict = training_set[i]  ## key : ['image', 'label', 'domain']
            for k, v in out_dict.items():
                if k in out_dicts.keys():
                    out_dicts[k].append(v)
                else:
                    out_dicts[k] = [v]
        
    all_indices = [all_indice[(gw * gh):] for all_indice in all_indices]
    for k,v in out_dicts.items():
        out_dicts[k] = np.stack(v)
    
    # get label from images with adapted pose-estimation network
    imgs = torch.tensor(out_dicts['image']).to(device).to(torch.float32) / 127.5 - 1
    with torch.no_grad():
        P.eval()
        _, label = P.get_pose_label(imgs)
        label = label.detach().to(device).to(torch.float32).detach().cpu().numpy() # batch x 25.
        P.train()
    out_dicts['label'] = label
    
    return out_dicts, all_indices, grid_indices

#----------------------------------------------------------------------------

def get_params_exclude_freeze(module, Nofreeze_keywords, rank, outdir):
    if rank == 0:
        print(f"Nofreeze_keywords: {Nofreeze_keywords}")

    target_param_names = []
    for keyword in Nofreeze_keywords:
        target_param_names.extend([name for name, _ in module.named_parameters() if name.find(keyword)>=0])
    target_param_names = list(set(target_param_names))
    # module_param_names = [name for name, _ in module.named_parameters() if name in target_param_names]
    # module_parameters = [p for n, p in module.named_parameters() if n in target_param_names]

    module_names_nofreeze = []
    module_names_freeze = []
    module_parameters = []
    for n, p in module.named_parameters():
        if n in target_param_names:
            module_parameters.append(p)
            module_names_nofreeze.append(n)
        else:
            module_names_freeze.append(n)

    if rank == 0:
        # print("="*200)
        # print("modules for optimization")
        # print("-"*200)
        # print(module_param_names)
        # print("="*200)
        with open(f"{outdir}nofreeze.txt", 'wt') as file:
            module_names_nofreeze = "\n".join(module_names_nofreeze)
            file.write(module_names_nofreeze)
        with open(f"{outdir}freeze.txt", 'wt') as file:
            module_names_freeze = "\n".join(module_names_freeze)
            file.write(module_names_freeze)

    return module_parameters

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    P_opt_kwargs            = {},       # Options for pose-estimation network optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    num_gpus                = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process in [0, num_gpus[.
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
    batch_gpu               = 4,        # Number of samples processed at a time by one GPU.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = 0.05,     # EMA ramp-up coefficient. None = no rampup.
    G_reg_interval          = None,     # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    # augment_p               = 0,        # Initial value of augmentation probability.
    # ada_target              = None,     # ADA target value. None = fixed p.
    # ada_interval            = 4,        # How often to perform ADA adjustment?
    # ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_kimg             = 0,        # First kimg to report when resuming training.
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    wandb_cfg               = {},       # Setting wandb
    from_args               = {},       # from argument
):
    # Initialize. ##########################################################
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False       # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = False             # Improves numerical accuracy.
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True                       # Improves training speed. # TODO: ENABLE
    grid_sample_gradfix.enabled = False                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')

    data_paths = training_set_kwargs.hparams.path
    num_domains = len(data_paths) if isinstance(data_paths, list) else 1
    domains = [i for i in range(num_domains)]
    if num_domains > 1 and from_args.use_balance_batch:
        assert batch_size // num_gpus % num_domains == 0
        training_sets = []
        training_set_samplers = []
        training_set_iterators = []
        for domain in domains:
            training_set_kwargs.hparams.use_domain_list = [domain]
            training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset
            training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
            training_sets.append(training_set)
            training_set_samplers.append(training_set_sampler)
            training_set_iterators.append(iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler,
                                                                     batch_size=batch_size // num_gpus // num_domains,
                                                                     **data_loader_kwargs)))
    else:
        training_sets = [dnnlib.util.construct_class_by_name(**training_set_kwargs)] # subclass of training.dataset.Dataset
        training_set_samplers = [misc.InfiniteSampler(dataset=training_sets[0], rank=rank, num_replicas=num_gpus, seed=random_seed)]
        training_set_iterators = [iter(torch.utils.data.DataLoader(dataset=training_sets[0], sampler=training_set_samplers[0], batch_size=batch_size//num_gpus, **data_loader_kwargs))]

    if rank == 0:
        print('Finish loading training set')
        print('Constructing networks...')

    # Construct networks.
    G = dnnlib.util.construct_class_by_name(**G_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()
    
    if from_args['run_Dr3D']:
        G_freeze = copy.deepcopy(G).eval()
        P = Posenet(pretrained_path=from_args['posenet_path']).train().requires_grad_(False).to(device)
        P_freeze = copy.deepcopy(P).eval()


    # Resume from existing pickle.
    save_resume_file = False
    if (resume_pkl is not None) and (rank == 0):
        try:
            with dnnlib.util.open_url(resume_pkl) as f:
                resume_data = legacy.load_network_pkl(f)
            print(f'Resuming from "{resume_pkl}"')
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
                misc.copy_params_and_buffers(resume_data[name], module, require_all=False, partial_copy=from_args['resume_partial_copy'])
            
            ## load posenet if pickle has.
            if 'P' in resume_data and from_args['run_Dr3D']:
                print("load pose network for resuming adaptation!!")
                misc.copy_params_and_buffers(resume_data['P'], P, require_all=False)
                misc.copy_params_and_buffers(resume_data['P'], P_freeze, require_all=False)
        except:
            ## load from different version
            new_resume_pkl = resume_pkl.replace(".pkl", "_")
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
                print(f'Resuming from "{new_resume_pkl}{name}.pth"')
                module.load_state_dict(torch.load(f"{new_resume_pkl}{name}.pth"))
            save_resume_file = True
        


    # Print network summary tables.
    if rank == 0:
        print('Finish Constructing networks')
        print(G)
        print(D)
        if from_args['run_Dr3D']:
            print(P)

    
    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    save_augment_img_path = None
    augment_pipes = [None for domain in domains]
    if (augment_kwargs is not None) and (augment_kwargs['augment_p'] > 0 or augment_kwargs['ada_target'] is not None):
        save_augment_img_path = f"{run_dir}/augment_img.png"
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_kwargs['augment_p']))
        if augment_kwargs['ada_target'] is not None:
            ada_stats = training_stats.Collector(regex='Loss_aug/signs/real')


    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    
    if from_args['run_Dr3D']:
        list_tmp = [G, D, G_ema, G_freeze, P, P_freeze, augment_pipe]
    else:
        list_tmp = [G, D, G_ema, augment_pipe]
    for module in list_tmp:
        if module is not None:
            if isinstance(module, list):
                for mod_ in module:
                    if mod_ is not None:
                        for param in misc.params_and_buffers(mod_):
                            if param.numel() > 0 and num_gpus > 1:
                                torch.distributed.broadcast(param, src=0)
            else:
                for param in misc.params_and_buffers(module):
                    if param.numel() > 0 and num_gpus > 1:
                        torch.distributed.broadcast(param, src=0)

    ## Save pretrain model loaded from different version
    if (save_resume_file) and (rank == 0):
        snapshot_pkl = resume_pkl.replace(".pkl", "_resave.pkl")
        if not os.path.exists(snapshot_pkl):
            print(f"Save resume file again : {snapshot_pkl}")
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if isinstance(module, list):
                    new_module = []
                    for i, mod_ in enumerate(module):
                        if mod_ is not None:
                            if num_gpus > 1:
                                misc.check_ddp_consistency(mod_, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                            new_module.append(copy.deepcopy(mod_).eval().requires_grad_(False).cpu())
                        else:
                            new_module.append(mod_)
                    snapshot_data[name] = new_module
                    del new_module
                    del mod_
                else:
                    if module is not None:
                        if num_gpus > 1:
                            misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                    snapshot_data[name] = module
                del module  # conserve memory
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)
            print("Done")
            return True
        else:
            print(f"It's already exist at {snapshot_pkl}")
            return False

    # Setup training phases. ######################################################
    if rank == 0:
        print('Setting up training phases...')
    if from_args['run_Dr3D']:
        loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, G_freeze=G_freeze, P=P, P_freeze=P_freeze, augment_pipe=augment_pipe, **loss_kwargs) # subclass of training.loss.Loss
    else:
        loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipes=augment_pipes, **loss_kwargs) # subclass of training.loss.Loss
    phases = []

    if from_args['run_Dr3D']:
        list_tmp = [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval), ('P', P, P_opt_kwargs, None)]
    else:
        list_tmp = [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]
    for name, module, opt_kwargs, reg_interval in list_tmp:
        if reg_interval is None:
            if name == 'P':
                lr = opt_kwargs['lr']
                opt = torch.optim.Adam([{'params': get_ignored_params(module.hopenet), 'lr': 0},
                                        {'params': get_non_ignored_params(module.hopenet), 'lr': lr},
                                        {'params': get_fc_params(module.hopenet), 'lr': lr * 5}],
                                        lr = lr)
            elif from_args['freeze_generator'] and name == 'G':
                params = get_params_exclude_freeze(module, from_args['Nofreeze_keywords'], rank, outdir=f"{run_dir}/{name}_")
                opt = dnnlib.util.construct_class_by_name(params=params, **opt_kwargs) # subclass of torch.optim.Optimizer
            else:
                opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            if from_args['freeze_generator'] and name == 'G':
                params = get_params_exclude_freeze(module, from_args['Nofreeze_keywords'], rank, outdir=f"{run_dir}/{name}_")
                opt = dnnlib.util.construct_class_by_name(params=params, **opt_kwargs) # subclass of torch.optim.Optimizer
            else:
                opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            # opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images. #############################################################
    if rank == 0:
        wandb_num_fixed_img = np.min([batch_gpu, wandb_cfg.wandb_num_fixed_img])
        if from_args['run_Dr3D']:
            fixed_out_dict, rnd_indices, fixed_indices = setup_snapshot_image_grid_Dr3D(training_sets=training_sets, all_indices=[[] for _ in domains], fix_indices=[[] for _ in domains], P=P, rank=rank, gw=1, gh=wandb_num_fixed_img//num_domains)
        else:
            fixed_out_dict, rnd_indices, fixed_indices = setup_snapshot_image_grid(training_sets=training_sets, all_indices=[[] for _ in domains], fix_indices=[[] for _ in domains], gw=1, gh=wandb_num_fixed_img//num_domains)
        G_ema.set_fixed_condition(fixed_out_dict, rank)

    # Initialize logs. ###################################################
    if rank == 0:
        print('Initializing logs...')
        if wandb_cfg.wandb_online is not True:
            os.environ["WANDB_MODE"]="offline"
        wandb_logger = wandb_cfg.wandb_logger
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')

    # Train. ###########################################################
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    while True:
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            out_dicts_list = [next(t) for t in training_set_iterators]  ## key : ['image', 'label', 'domain']
            out_dicts = out_dicts_list[0]
            for out_dict in out_dicts_list[1:]:
                for key in out_dicts:
                    out_dicts[key] = torch.cat((out_dicts[key], out_dict[key]), dim=0)
            
            
            if from_args['run_Dr3D']:
                real_inputs, gen_inputs, num_batch = G.get_input(out_dicts, training_sets, rank, batch_gpu, batch_size, phases, P)
            else:
                real_inputs, gen_inputs, num_batch = G.get_input(out_dicts, training_sets, rank, batch_gpu, batch_size, phases)

        # Execute training phases.
        for p_idx, phase in enumerate(phases):
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            # Accumulate gradients.
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for n in range(num_batch):
                phase_inputs = {}
                for k, v in real_inputs.items():
                    phase_inputs[k] = v[n]
                for k, v in gen_inputs.items():
                    phase_inputs[k] = v[p_idx][n]
                if phase.name == 'Dmain':
                    loss.accumulate_gradients(phase=phase.name, inputs=phase_inputs, gain=phase.interval, cur_nimg=cur_nimg, save_augment_img_path=save_augment_img_path)
                    save_augment_img_path = None
                else:
                    loss.accumulate_gradients(phase=phase.name, inputs=phase_inputs, gain=phase.interval,
                                              cur_nimg=cur_nimg, save_augment_img_path=None)
            phase.module.requires_grad_(False)

            # Update weights.
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.numel() > 0 and param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            # Phase done.
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
            G_ema.neural_rendering_resolution = G.neural_rendering_resolution
            G_ema.rendering_kwargs = G.rendering_kwargs.copy()

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1
        global_step = int(cur_nimg / 1e3)

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % augment_kwargs['ada_interval'] == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss_aug/signs/real'] - augment_kwargs['ada_target']) * (batch_size * augment_kwargs['ada_interval']) / (augment_kwargs['ada_kimg'] * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))
            

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(f"step : {global_step}")
            #print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):

            wandb_num_rnd_img = np.min([batch_gpu, wandb_cfg.wandb_num_rnd_img])
            if from_args['run_Dr3D']:
                rnd_out_dict, rnd_indices, _ = setup_snapshot_image_grid_Dr3D(training_sets=training_sets,
                                                                     all_indices=rnd_indices,
                                                                     fix_indices=fixed_indices,
                                                                     P=P,
                                                                     rank=rank,
                                                                    gw=1, gh=wandb_num_rnd_img//num_domains)
            else:
                rnd_out_dict, rnd_indices, _ = setup_snapshot_image_grid(training_sets=training_sets,
                                                                     all_indices=rnd_indices,
                                                                     fix_indices=fixed_indices,
                                                                    gw=1, gh=wandb_num_rnd_img//num_domains)
            
            G_ema.set_random_condition(rnd_out_dict, rank)
            log_dict = G_ema.synthesis_images_for_display(global_step, print_angle=from_args['print_angle'])
            wandb_logger.log_images(log_dict, global_step)
            torch.cuda.empty_cache()

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            if from_args['run_Dr3D']:
                list_tmp = [('G', G), ('D', D), ('G_ema', G_ema), ('P', P),  ('augment_pipe', augment_pipe)]
            else:
                list_tmp = [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]
            for name, module in list_tmp:
                if isinstance(module, list):
                    new_module = []
                    for i, mod_ in enumerate(module):
                        if mod_ is not None:
                            if num_gpus > 1:
                                misc.check_ddp_consistency(mod_, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                            new_module.append(copy.deepcopy(mod_).eval().requires_grad_(False).cpu())
                        else:
                            new_module.append(mod_)
                    snapshot_data[name] = new_module
                    del new_module
                    del mod_
                else:
                    if module is not None:
                        if num_gpus > 1:
                            misc.check_ddp_consistency(module, ignore_regex=r'.*\.[^.]+_(avg|ema|mean|var)')
                        module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                    snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{global_step:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # # Evaluate metrics.
        # if (snapshot_data is not None) and (len(metrics) > 0):
        #     if rank == 0:
        #         print(run_dir)
        #         print('Evaluating metrics...')
        #     for metric in metrics:
        #         result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
        #             dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank, device=device)
        #         if rank == 0:
        #             metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
        #         stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()

        if rank==0:
            for name, value in stats_dict.items():
                stats_dict[name] = value.mean
            wandb_logger.log_dicts(stats_dict, global_step)
            wandb_logger.log_dicts(stats_metrics, global_step)
        if progress_fn is not None:
            progress_fn(global_step, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

    return False
#----------------------------------------------------------------------------
