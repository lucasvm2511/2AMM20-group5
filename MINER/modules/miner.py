#!/usr/bin/env python

import pickle
import os
import sys
import tqdm
import time
import pdb
import copy
import configparser
import argparse

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import cv2
import torch
from torch.nn import Parameter

import folding_utils as unfoldNd
import nvidia_smi

import matplotlib.pyplot as plt
plt.gray()

import losses
import utils
import volutils
import siren

def load_config(configfile):
    '''
        Load configuration file
    '''
    config_dict = configparser.ConfigParser()
    config_dict.read(configfile)
    
    config = argparse.Namespace()
    
    # Specific modifications
    config.ksize = int(config_dict['FOLDING']['ksize'])
    config.stride = int(config_dict['FOLDING']['stride'])
    
    config.in_features = int(config_dict['NETWORK']['in_features'])
    config.out_features = int(config_dict['NETWORK']['out_features'])
    config.nfeat = int(config_dict['NETWORK']['nfeat'])
    config.nlayers = int(config_dict['NETWORK']['nlayers'])
    config.omega_0 = float(config_dict['NETWORK']['omega_0'])
    config.nonlin = config_dict['NETWORK']['nonlin']
    
    config.lr = float(config_dict['TRAINING']['lr'])
    config.loss = config_dict['TRAINING']['loss']
    config.epochs = int(config_dict['TRAINING']['epochs'])
    config.switch_thres = float(config_dict['TRAINING']['switch_thres'])
    config.weightshare = config_dict['TRAINING']['weightshare']
    config.propmethod = config_dict['TRAINING']['propmethod']
    config.coordstype = config_dict['TRAINING']['coordstype']
    config.maxchunks = int(config_dict['TRAINING']['maxchunks'])
    
    return config

def miner_imfit(im, nscales, switch_mse, stopping_mse, config, visualize_image, visualization_channels = None):
    '''
        Multiscale Neural Implicit Representation (MINER) fitting for images
        
        Inputs:
            im: (H, W, 3) Image to fit. 
            nscales: Number of scales to fit
            switch_mse: Blocks are terminated when this MSE is achieved
            stopping_mse: Fitting is halted if this MSE is achieved. Set to 0
                to run for all iterations
            config: Namespace with configuration variables. Check configs folder
                for more examples
                
        Outputs:
            imfit: Final fitted image
            info: Dictionary with debugging information
                mse_array: MSE as a function of epochs
                time_array: Time as a function of epochs
                num_units_array: Number of active units as a function of epochs
                learn_indices_list: Active indices at the starting of each
                    scale of fitting
                nparams: Total number of parameters
                
        TODO: Return all model parameters
        WARNING: Fitting beyond 4096x4096 is currently not implemented
    '''
    H, W, _ = im.shape
    # if sys.platform == 'win32':
    visualize = visualize_image
    viewsize = (1024, (H*1024)//W)
    # else:
    #     visualize = False
        
    if (H*W < 4096*8192):
        full_compare = True
    else:
        full_compare = False

    # Send image to CUDA
    imten = torch.tensor(im).permute(2, 0, 1)
    
    if full_compare:
        imten_gpu = imten.cuda()
    
    # Show ground truth image
    if visualize:

        if visualization_channels:
            cv2.imshow('GT', cv2.resize(im[:, :, visualization_channels], viewsize, interpolation=cv2.INTER_AREA))
        else:
            cv2.imshow('GT', cv2.resize(im, viewsize, interpolation=cv2.INTER_AREA))
        cv2.waitKey(1)
    
    # Parameters to save across iterations
    learn_indices_list = []
    nparams = 0
    mse_array = np.zeros(config.epochs*(nscales+1))
    time_array = np.zeros(config.epochs*(nscales+1))
    num_units_array = np.zeros(config.epochs*(nscales+1))
    nparams_array = np.zeros(nscales)
    memory_array = np.zeros(nscales)
    psnr_array = np.zeros(nscales)
    
    prev_params = None
    im_estim = None
        
    # Memory usage helpers -- WARNING -- GPU ID is hardcoded
    if sys.platform != 'win32':
        nvidia_smi.nvmlInit()
        mem_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    
    # Begin
    for scale_idx in tqdm.tqdm(range(nscales), dynamic_ncols=True):
        imtarget_ten = torch.nn.functional.interpolate(imten[None, ...],
                                    scale_factor=pow(2.0, -nscales+scale_idx+1),
                                    mode='area')
        _, _, Ht, Wt = imtarget_ten.shape
        
        # Create folders and unfolders
        unfold = torch.nn.Unfold(kernel_size=config.ksize, stride=config.stride)
        fold = torch.nn.Fold(output_size=(Ht, Wt), kernel_size=config.ksize,
                            stride=config.stride)
   
        # Find out number of chunks
        nchunks = int(Ht*Wt/(config.ksize**2))   
        
        # Create model
        if scale_idx == 0:
            nfeat = config.nfeat*4
            lr_div = 4
        else:
            nfeat = config.nfeat
            lr_div = 1
                            
        # Create inputs
        coords_chunked = get_coords((Ht, Wt), config.ksize,
                                    config.coordstype, unfold)
                            
        imten_chunked = unfold(imtarget_ten.permute(1, 0, 2, 3))
        
        criterion = losses.L2Norm()
        criterion_tv = losses.TVNorm()
        best_mse = float('inf')
        best_img = None
            
        tbar = tqdm.tqdm(range(config.epochs), dynamic_ncols=True)
        
        ret_list = get_next_signal(imtarget_ten, scale_idx, config, unfold, 
                                   switch_mse, nchunks, im_estim)
        [imten_chunked, im_chunked_prev, im_chunked_res_frozen, 
                master_indices, learn_indices, loss_frozen] = ret_list
        
        learn_indices_list.append(master_indices)
                
        # Make a copy of the parameters and then truncate  
        if scale_idx < 2:      
            model, prev_params = get_model(config, nchunks, nfeat,
                                           master_indices, None)
        else:
            model, prev_params = get_model(config, nchunks, nfeat,
                                           master_indices, prev_params)
                                
        # Truncate coordinates and training data
        coords_chunked = coords_chunked[master_indices, ...]
        imten_chunked = imten_chunked[..., master_indices]
                    
        # Optimizer
        optim = torch.optim.Adam(lr=config.lr/lr_div,
                                 params=model.parameters())
                
        nparams += utils.count_parameters(model)
        nparams_array[scale_idx] = utils.count_parameters(model)
                
        prev_loss = 0
        signal_norm = (imten_chunked**2).mean(0, keepdim=True)
        signal_norm = signal_norm.mean(1, keepdim=True).flatten()
        
        # Move tensors to GPU
        coords_chunked = coords_chunked.cuda(non_blocking=True)
        imten_chunked = imten_chunked.cuda(non_blocking=True)
        im_chunked_prev = im_chunked_prev.cuda(non_blocking=True)
        im_chunked_res_frozen = im_chunked_res_frozen.cuda(non_blocking=True)
        learn_indices = learn_indices.cuda(non_blocking=True)
        master_indices = master_indices.cuda(non_blocking=True)
        signal_norm = signal_norm.cuda(non_blocking=True)
        
        for idx in tbar:             
            if learn_indices.numel() == 0:
                break
            lr = config.lr*learn_indices.numel()/master_indices.numel()
            weight = pow(0.999, idx)
            optim.param_groups[0]['lr'] = lr*weight
            
            if learn_indices.numel() > config.maxchunks:
                im_sub_chunked_list = []
                npix = config.ksize**2
                im_sub_chunked = torch.zeros((3, npix, learn_indices.numel()),
                                             device='cuda')
                lossval = 0
                weight = 0
                for sub_idx in range(0, learn_indices.numel(), config.maxchunks):
                    sub_idx2 = min(learn_indices.numel(),
                                   sub_idx+config.maxchunks)
                    learn_indices_sub = learn_indices[sub_idx:sub_idx2]
                    
                    optim.zero_grad()
            
                    im_sub_chunked_sub = model(coords_chunked,
                                            learn_indices_sub).permute(2, 1, 0)
                        
                    loss = criterion(im_sub_chunked_sub,
                                     imten_chunked[..., learn_indices_sub]) 
                    
                    loss.backward()                    
                    optim.step()
                
                    lossval += loss.item()    
                    weight += 1
                    
                    with torch.no_grad():
                        im_sub_chunked_list.append(im_sub_chunked_sub)
                        
                lossval /= weight
                with torch.no_grad():
                    im_sub_chunked = torch.cat(im_sub_chunked_list, 2)

            else:
                optim.zero_grad()
                
                im_sub_chunked = model(coords_chunked,
                                      learn_indices).permute(2, 1, 0)
                loss = criterion(im_sub_chunked,
                            imten_chunked[..., learn_indices])
                
                loss.backward()
                optim.step()                
                lossval = loss.item()
                
            with torch.no_grad():
                chunk_err = im_sub_chunked - imten_chunked[..., learn_indices]
                
                chunk_err = (chunk_err**2).mean(1, keepdim=True)
                chunk_err = chunk_err.mean(0, keepdim=True)
                
                # Freeze sections that have achieved target MSE
                learn_indices_sub, = torch.where(chunk_err.flatten() > \
                    switch_mse*signal_norm[learn_indices])
                
                new_indices = master_indices[learn_indices[learn_indices_sub]]
                im_chunked_res_frozen[..., new_indices] = \
                    im_sub_chunked[..., learn_indices_sub]
                    
                if config.propmethod == 'coarse2fine':
                    im_estim = fold(im_chunked_res_frozen +\
                        im_chunked_prev)[:, 0, ...]
                else:
                    im_estim = fold(im_chunked_res_frozen)[:, 0, ...]
                                
                if scale_idx < nscales - 1:
                    if full_compare:
                        im_estim_up = torch.nn.functional.interpolate(
                            im_estim[None, ...],
                            size=(H, W),
                            mode='bilinear'
                        )[0, ...]
                        mse_full = ((im_estim_up - imten_gpu)**2).mean()
                    else:
                        mse_full = 1
                else:
                    mse_full = (lossval*learn_indices.numel() + \
                                loss_frozen)/nchunks
                        
            # Update indices to learn
            learn_indices = learn_indices[learn_indices_sub]
            
            if scale_idx < nscales - 1:
                if abs(lossval - prev_loss) < config.switch_thres:
                    break
            prev_loss = lossval
            
            if full_compare:
                descloss = mse_full
            else:
                if scale_idx == nscales - 1:
                    descloss = mse_full
                else:
                    descloss = lossval
                
            mse_array[scale_idx*config.epochs + idx] = descloss
            time_array[scale_idx*config.epochs + idx] = time.time()
            
            num_units_array[scale_idx*config.epochs + idx] = learn_indices.numel()
            
            if lossval < best_mse:
                best_mse = lossval
                best_img = copy.deepcopy(im_estim.detach())
                
            if mse_full < stopping_mse:
                break
                
            if visualize:
                im_estim_cpu = im_estim.detach().cpu().permute(1, 2, 0).numpy()

                if visualization_channels:
                    cv2.imshow('Estim', cv2.resize(im_estim_cpu[:, :, visualization_channels], viewsize, interpolation=cv2.INTER_AREA))
                else:
                    cv2.imshow('Estim', cv2.resize(im_estim_cpu, viewsize, interpolation=cv2.INTER_AREA))

                cv2.waitKey(1)
            
            # Memory usage -- WARNING: Implemented only for Linux
            if sys.platform != 'win32':
                mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(mem_handle)
                mem_usage = mem_info.used/2**30
            else:
                mem_usage = 0.0
            
            if idx == 0:
                memory_array[scale_idx] = mem_usage
            
            tbar.set_description('[%d/%d | %.2f GB| %.4e]'%(
                learn_indices.numel(),
                nchunks,
                mem_usage,
                descloss))
            tbar.refresh()
        
        if scale_idx < nscales - 1:
            # Copy the model, we will need it to propagate parameters
            prev_params_sub = copy.deepcopy(model.state_dict())
            
            # First copy the new parameters
            for key in prev_params:
                prev_params[key][master_indices, ...] = prev_params_sub[key][...]
            
            # Now we will double up all the parameters
            indices = np.arange(nchunks).reshape(Ht//config.ksize,
                                                 Wt//config.ksize)
            indices = cv2.resize(indices, None, fx=2, fy=2,
                                interpolation=cv2.INTER_NEAREST).ravel()
            indices = torch.tensor(indices).cuda().long()
            
            for key in prev_params:
                prev_params[key] = prev_params[key][indices, ...]/2.0
                                       
        # Move tensors to CPU
        im_estim = im_estim.cpu().squeeze()
        scale_factor = 2 ** (nscales - scale_idx - 1)

        psnr_array[scale_idx] = utils.psnr(im, im_estim.permute(1, 2, 0).numpy().repeat(scale_factor,axis=0).repeat(scale_factor,axis=1))

        if visualization_channels:
            save_img = np.clip(im_estim.permute(1, 2, 0)[:,:,visualization_channels].numpy(), 0, 1)
        else:
            save_img = np.clip(im_estim.permute(1, 2, 0).numpy(), 0, 1)


        cv2.imwrite('%s/scale%d.png'%(config.savedir, scale_idx),
                   (255*save_img).astype(np.uint8))
        
        imblock = drawblocks_single(master_indices.cpu(), (Ht, Wt),
                                    (config.ksize, config.ksize))
        imblock = imblock.cpu().numpy()

        if visualization_channels:
            save_img = np.clip(im_estim.permute(1, 2, 0)[:, :, visualization_channels].numpy(), 0, 1)
        else:
            save_img = np.clip(im_estim.permute(1, 2, 0).numpy(), 0, 1)

        save_img = np.clip(save_img + imblock, 0, 1)
        cv2.imwrite('%s/scale%d_ann.png'%(config.savedir, scale_idx),
                   (255*save_img).astype(np.uint8))

        # Save model
        model_save_path = os.path.join(config.savedir, f"scale{scale_idx}_model_state.pt")
        torch.save(model.state_dict(), model_save_path)

        torch.cuda.empty_cache()
                        
    # Remove zero valued entries
    indices, = np.where(time_array > 0)
    
    info = {'time_array': time_array[indices],
            'mse_array': mse_array[indices],
            'num_units_array': num_units_array[indices],
            'nparams': nparams,
            'nparams_array': nparams_array,
            'memory_array': memory_array,
            'learn_indices_list': learn_indices_list,
            'psnr_array' : psnr_array}
    
    return best_img.cpu().permute(1, 2, 0).numpy(), info
  
def drawblocks(learn_indices_list, imsize, ksize):
    '''
        Draw lines around blocks to showcase when they got terminated
        
        Inputs:
            learn_indices_list: List of tensor arrarys of indices
            imsize: Size of image            
            ksize: Size of kernel
            
        Outputs:
            im_labels: Labeled image
    '''
    im_labels = torch.zeros((1, 1, imsize[0], imsize[1]))
    nscales = len(learn_indices_list)
    H, W = imsize
    
    for idx in range(len(learn_indices_list)):
        learn_indices = learn_indices_list[idx]
        
        fold_tsize = (ksize*pow(2, nscales-idx-1),
                      ksize*pow(2, nscales-idx-1))
        
        # Create folders and unfolders
        unfold = torch.nn.Unfold(kernel_size=fold_tsize, stride=fold_tsize)
        fold = torch.nn.Fold(output_size=(H, W), kernel_size=fold_tsize,
                            stride=fold_tsize)
        
        im_labels_chunked = unfold(im_labels).reshape(1, fold_tsize[0],
                                                      fold_tsize[1], -1)
        im_labels_chunked[:, 0:2, :, learn_indices] = idx + 1
        im_labels_chunked[:, -3:-1, :, learn_indices] = idx + 1
        im_labels_chunked[:, :, 0:2, learn_indices] = idx + 1
        im_labels_chunked[:, :, -3:-1, learn_indices] = idx + 1
        
        im_labels_chunked = im_labels_chunked.reshape(1,
                                                      fold_tsize[0]*fold_tsize[1],
                                                      -1)
        im_labels = fold(im_labels_chunked)       
        
    return im_labels.squeeze().detach().cpu()

def drawblocks_single(learn_indices, imsize, ksize):
    '''
        Draw blocks for a single image at a single scale
        
        Inputs:
            learn_indices: List of active indices
            imsize: Size of the image
            ksize: Kernel size
            
        Outputs:
            im_labels: Label image
    '''
    im_labels = torch.zeros((3, 1, imsize[0], imsize[1]))
    
    # Create folders and unfolders
    unfold = torch.nn.Unfold(kernel_size=ksize, stride=ksize)
    fold = torch.nn.Fold(output_size=imsize, kernel_size=ksize,
                        stride=ksize)
    
    im_labels_chunked = unfold(im_labels).reshape(3, ksize[0], ksize[1], -1)
    
    im_labels_chunked[2, 0, :, learn_indices] = 1
    im_labels_chunked[2, -1, :, learn_indices] = 1
    im_labels_chunked[2, :, 0, learn_indices] = 1
    im_labels_chunked[2, :, -1, learn_indices] = 1
    
    im_labels_chunked[1, 0, :, learn_indices] = 1
    im_labels_chunked[1, -1, :, learn_indices] = 1
    im_labels_chunked[1, :, 0, learn_indices] = 1
    im_labels_chunked[1, :, -1, learn_indices] = 1
    
    im_labels_chunked = im_labels_chunked.reshape(3, ksize[0]*ksize[1], -1)
    im_labels = fold(im_labels_chunked).permute(2, 3, 0, 1).squeeze()
        
    return im_labels

def drawcubes_single(learn_indices, cubesize, ksize, savename):
    '''
    Draw blocks for a single cube at a single scale
    
    Inputs:
        learn_indices: Indices for drawing the blocks
        cubesize: Size of the full cube
        ksize: Size of folding kernel
        savename: Name of the file for saving the plot
        
    Outputs:
        None
    '''
    unfold = unfoldNd.UnfoldNd(kernel_size=ksize, stride=ksize)
    
    # Get coordinates 
    coords_chunked = get_coords(cubesize, ksize, 'global', unfold)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('auto')
    
    size = 2*np.array(ksize)/np.array(cubesize)
    size[...] = size[0]
    for idx in learn_indices:
        pos, _ = coords_chunked[idx, :, :].min(0)
        volutils.plotCubeAt(pos=pos[[0, 1, 2]], size=size, color='tab:blue',
                            edgecolor='tab:blue', alpha=0.05, ax=ax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.set_zticks([-1, 1])
    
    plt.savefig(savename)
    plt.close('all')

def get_coords(imsize, ksize, coordstype, unfold):
    '''
        Generate coordinates for MINER training
        
        Inputs:
            imsize: (H, W) image size
            ksize: Kernel size
            coordstype: 'global' or 'local'
            unfold: Unfold operator
    '''
    ndim = len(imsize)
    if ndim == 2:
        H, W = imsize    
        nchunks = int(H*W/(ksize**ndim))
        # Create inputs
        if coordstype == 'global':
            X, Y = torch.meshgrid(torch.linspace(-1, 1, W),
                                  torch.linspace(-1, 1, H))
            coords = torch.cat((X[None, None, ...], Y[None, None, ...]), 0)
            coords_chunked = unfold(coords).permute(2, 1, 0)
        elif coordstype == 'local':
            Xsub, Ysub = torch.meshgrid(torch.linspace(-1, 1, ksize),
                                        torch.linspace(-1, 1, ksize))
            coords_sub = torch.cat((Xsub[None, None, ...],
                                    Ysub[None, None, ...]), 0)
            coords_chunked_sub = unfold(coords_sub).permute(2, 1, 0)
            coords_chunked = torch.repeat_interleave(coords_chunked_sub,
                                                     nchunks, 0)
        else:
            raise AttributeError('Coordinate type not understood')
    else:
        H, W, T = imsize    
        nchunks = int(H*W*T/(ksize**ndim))
        # Create inputs
        if coordstype == 'global':
            X, Y, Z = torch.meshgrid(torch.linspace(-1, 1, W),
                                     torch.linspace(-1, 1, H),
                                     torch.linspace(-1, 1, T))
            coords = torch.cat((X[None, None, ...],
                                Y[None, None, ...],
                                Z[None, None, ...]), 0)
            coords_chunked = unfold(coords).permute(2, 1, 0)
        elif coordstype == 'local':
            Xsub, Ysub, Zsub = torch.meshgrid(torch.linspace(-1, 1, ksize),
                                              torch.linspace(-1, 1, ksize),
                                              torch.linspace(-1, 1, ksize))
            coords_sub = torch.cat((Xsub[None, None, ...],
                                    Ysub[None, None, ...],
                                    Zsub[None, None, ...]), 0)
            coords_chunked_sub = unfold(coords_sub).permute(2, 1, 0)
            coords_chunked = torch.repeat_interleave(coords_chunked_sub,
                                                     nchunks, 0)
        else:
            raise AttributeError('Coordinate type not understood')
    
    return coords_chunked

@torch.no_grad()
def get_model(config, nchunks, nfeat, master_indices=None, prev_params=None,
              scale_idx=0):
    '''
        Get an adaptive kilo-siren
        
        Inputs:
            config: Configuration structure
            master_indices: Indices used for copying previous parameters
                If None, previous parameters are not copied
            prev_params: If master_indices is not None, then these parameters
                are copied to the model.
            scale_idx: Scale of fitting
    '''        
    # Create a model first
    #if config.signaltype == 'occupancy':
    if config.loss == 'logistic':
        const = 2.0
    else:
        const = pow(2, -scale_idx)
    model = siren.AdaptiveMultiSiren(in_features=config.in_features,
                                    out_features=config.out_features, 
                                    n_channels=nchunks,
                                    hidden_features=nfeat, 
                                    hidden_layers=config.nlayers,
                                    outermost_linear=True,
                                    first_omega_0=config.omega_0,
                                    hidden_omega_0=config.omega_0,
                                    nonlin=config.nonlin,
                                    const=const).cuda()
    
    if config.weightshare == 'noshare':
        prev_params = copy.deepcopy(model.state_dict())
    
    # Load, save, truncate
    if prev_params is not None:
        model.load_state_dict(prev_params)
    else:
        prev_params = copy.deepcopy(model.state_dict())
        
    # Then see if we need to copy parameters
    if prev_params is not None:
        for l_idx in range(len(model.net)-1):
            # N-1 layers are with non linearity
            model.net[l_idx].linear.weight = \
                Parameter(model.net[l_idx].linear.weight[master_indices, ...])
                
            model.net[l_idx].linear.bias = \
                Parameter(model.net[l_idx].linear.bias[master_indices, ...])
        
        # Last layer is just linear
        model.net[-1].weight = \
            Parameter(model.net[-1].weight[master_indices, ...])
            
        model.net[-1].bias = \
            Parameter(model.net[-1].bias[master_indices, ...])
                
    return model, prev_params

@torch.no_grad()
def get_next_signal(imtarget_ten, scale_idx, config, unfold, switch_mse,
                    nchunks, im_estim=None, ndim=2):
    '''
        Get the signal for next scale
        
        Inputs:
            imtarget_ten: (1, 3, H, W) image to fit
            scale_idx: Scale of fitting
            config: Configuration structure
            unfold: Unfolding operator
            switch_mse: Threshold operator
            im_estim: estimate from previous fitting. Ignored if scale_idx is 0
            ndim: Number of dimensions of the signal. 2 for images and 3 for
                volumetric data
    '''
    if ndim == 2:
        imten_chunked = unfold(imtarget_ten.permute(1, 0, 2, 3))
        mode = 'bilinear'
    else:
        imten_chunked = unfold(imtarget_ten)
        mode = 'trilinear'
    
    if scale_idx == 0:
        # For first scale, we need to use up all indices. Next scale is
        # decided by error from upsampling
        learn_indices = torch.arange(nchunks)
        master_indices = torch.arange(nchunks)
        
        nchan = imtarget_ten.shape[1]
    
        im_chunked_prev = torch.zeros(nchan, config.ksize**ndim, nchunks)
        im_chunked_res_frozen = torch.zeros(nchan, config.ksize**ndim, nchunks)
        
        if config.signaltype == 'occupancy':
            loss_frozen = [0, 1]
        else:
            loss_frozen = 0
        freeze_indices = torch.tensor([]).long()
    else:
        # Upsample previous reconstruction
        im_estim_prev = torch.nn.functional.interpolate(im_estim[None, ...], 
                                                        scale_factor=2,
                                                        mode=mode)
        
        if ndim == 2:
            im_chunked_prev = unfold(im_estim_prev.permute(1, 0, 2, 3))
        else:
            im_chunked_prev = unfold(im_estim_prev)
        
        # Compute reside -- we will fit to this instead of whole signal
        try:
            im_chunked_res_frozen = imten_chunked - im_chunked_prev
        except RuntimeError:
            pdb.set_trace()
                        
        # If the energy in residue is very small, do not learn anything
        # over that patch
        #if config.signaltype == 'occupancy':
        if config.loss == 'logistic':
            criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
            chunk_err = criterion(im_chunked_prev, imten_chunked)
            chunk_err = chunk_err.mean(1, keepdim=True)
        else:
            chunk_err = (im_chunked_res_frozen**2).mean(1, keepdim=True)
            #chunk_err = im_chunked_prev.std(1, keepdim=True)
        chunk_err = chunk_err.mean(0, keepdim=True)
        
        master_indices, = torch.where(chunk_err.flatten() > switch_mse)
        freeze_indices, = torch.where(chunk_err.flatten() <= switch_mse)
        
        thres = 0.0005
        master_indices, = torch.where(chunk_err.flatten() > thres)
        freeze_indices, = torch.where(chunk_err.flatten() <= thres)
        
        if config.signaltype == 'occupancy':
            loss_frozen = [0, 1]
        else:
            loss_frozen = chunk_err.flatten()[freeze_indices].sum()
        
        # Now we need to get back only the coarse coordinates
        im_chunked_res_frozen[..., freeze_indices] = 0
        
        # Change chunked tensor to residual if fitting residues
        if config.propmethod == 'coarse2fine':
            imten_chunked = im_chunked_res_frozen.clone()
        else:
            # We continue fitting to signal instead of residue
            im_chunked_res_frozen = im_chunked_prev
                    
        # Since we are truncating the model, we will have learn
        # indices as all 1 to numel
        if master_indices.numel() == 0:
            # Nominal number of blocks
            master_indices = torch.tensor([0, 1])
            learn_indices = learn_indices = torch.tensor([0, 1])
        else:
            learn_indices = torch.arange(master_indices.numel())
        
    ret_list = [imten_chunked, im_chunked_prev, im_chunked_res_frozen, 
                master_indices, learn_indices, loss_frozen] 
        
    return ret_list