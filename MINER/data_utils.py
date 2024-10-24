import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import modules.utils as utils
import modules.miner as miner
import time
import cv2
import pickle
from skimage.metrics import structural_similarity as ssim_func
from matplotlib import cm
plt.gray()
import warnings
warnings.filterwarnings("ignore")
import copy

class SatelliteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self._root_dir = root_dir
        self._files = [os.path.join(root_dir, f) 
                       for f in os.listdir(root_dir)]
        
        self._transform = transform

    def __len__(self):
        return len(self._tif_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.load(self._files[idx])

        img_tensor = torch.from_numpy(img.astype(np.float32))

        if self._transform:
            img_tensor = self._transform(img_tensor)

        return img_tensor

class Sentinel2Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self._root_dir = root_dir
        self._location_types = [f for f in os.listdir(root_dir) if os.path.isdir(f)]
        if len(self._location_types) == 0:
            self._location_types = [""]
        self._tif_files = [
            os.path.join(self._root_dir, drct, img)
            for drct in self._location_types
            for img in os.listdir(os.path.join(self._root_dir, drct))
            if img.endswith('.tif')
        ]

        
        self._transform = transform

    def __len__(self):
        return len(self._tif_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        with rasterio.open(self._tif_files[idx]) as src:
            img = src.read()  # This reads the image as a (channels, height, width) numpy array

        img_tensor = torch.from_numpy(img.astype(np.float32))

        if self._transform:
            img_tensor = self._transform(img_tensor)

        return img_tensor
    

TRUE_COLOR = [3, 2, 1]
FALSE_COLOR = [7, 3, 2]
AGRICULTURE = [11,8,2]

def viewImage(tensor, channels = TRUE_COLOR):

    rgbvals = tensor[:, :, channels]
    max_val = torch.max(rgbvals)
    rgbvals_norm = np.array(rgbvals / max_val * 255, dtype = int)
    plt.imshow(rgbvals_norm)
    plt.show()

def read_config(config_type, nscales = 2):
    """
    Creates a configuration dictionary from an ini file.

    configname      : Configuration file name
    nscales         : Number of MINER scales
    """

    configuration_names = {
        "image"         : "configs/image_32x32.ini",
        "kaggle"        : "configs/tensor_8x8_13.ini",
        "satellite"     : "configs/satellite.ini",
        "satellite_rgb" : "configs/satellite_rgb.ini" 
    }
    
    # Read configuration
    config = miner.load_config(configuration_names[config_type])
    config.signaltype = 'image'
    config.config_type = config_type
    config.nscales = nscales

    return config

def readImage(image_name, config):
    """
    Reads an image and transforms it into the correct format.

    image_name      : Image name (jpg)
    config          : Configuration dictionary
    """

    # Load image
    im = cv2.imread(os.path.join('data', f'{image_name}.jpg')).astype(np.float32)/255.0
    config.image_name = image_name

    return transformArray(im, config)


def readTensor(tensor, index, config):
    """
    Reads a tensor and transforms it into the correct format.

    tensor      : Tensor
    index       : Index of tensor
    config      : Configuration dictionary
    """

    # Load tensor
    max_val = torch.max(tensor)
    im = (tensor/max_val).numpy().astype(np.float32) # TODO: Proper scaling
    config.image_name = f"{config.config_type}_{index}"

    return transformArray(im, config)
    

def transformArray(im, config):
    """
    Transforms a numpy array into the correct format.

    im     : Image (numpy array)
    config : Configuration dictionary
    """

    # Clipping image ensures there are a whole number of blocks
    clip_size = (config.stride*pow(2, config.nscales-1),
                 config.stride*pow(2, config.nscales-1))
    
    im = utils.moduloclip(im, clip_size)
    H, W, _ = im.shape
    
    config.savedir = os.path.join(os.getcwd(), "results", f'{config.image_name}_{W}_{H}')

    os.makedirs(config.savedir, exist_ok=True)

    return im

def get_model_size(config):
    model_states = [os.path.join(config.savedir, f) for f in os.listdir(config.savedir) if "model_state" in f]
    model_states.sort()
    total_bits_array = np.array(list(map(os.path.getsize, model_states))) * 8
    total_bits_cum = np.add.accumulate(total_bits_array)

    return total_bits_cum

def run_miner(image, 
              config,  
              stopping_mse = 1e-4, 
              target_mse = 1e-4, 
              visualize_image = False,
              verbose = True,
              save_results = True,
              visualization_channels = None):
    """
    Runs the MINER algorithm
    
    image           : Image to compress
    config          : Configuration dictionary
    stopping_mse    : When to stop image fitting
    target_mse      : Per-block stopping criterion
    visualize_image : Boolean to visualize estimations during training
    verbose         : Boolean to print output metrics
    save_results    : Boolean to save the results in a .mat file
    """
    
    torch.cuda.empty_cache()
    
    # Run MINER
    tic = time.time()
    best_im, info = miner.miner_imfit(image, 
                                      config.nscales,
                                        target_mse,
                                        stopping_mse,
                                        config,
                                        visualize_image,
                                        visualization_channels)
    total_time = time.time() - tic

    psnr_val = utils.psnr(image, best_im)

    if verbose:
        print('Total time %.2f minutes'%(total_time/60))
        print('PSNR: ', psnr_val)

        try:
            print('SSIM: ', ssim_func(image, best_im, multichannel=True))
        except:
            pass
        print('Total pararmeters: %.2f million'%(info["nparams"]/1e6))    


    if save_results:

        info["time_array"] -= info["time_array"][0]

        total_bits_cumulative_array = get_model_size(config)

        H, W, _ = best_im.shape
        
        os.makedirs('results', exist_ok=True)
        mdict = {'mse_array': info['mse_array'],
                'time_array': info['time_array'],
                'nparams': info['nparams'],
                'memory_array': info['memory_array'],
                'nparams_array': info['nparams_array'],
                'psnr_best_im': psnr_val,
                'psnr_array' : info['psnr_array'],
                'total_bits_cumulative_array' : total_bits_cumulative_array,
                'bits_per_pixel_array' : total_bits_cumulative_array / (H * W)}
        
        save_path = os.path.join(config.savedir, "metrics.pkl")

        with open(save_path, 'wb') as f:
            pickle.dump(mdict, f)

    torch.cuda.empty_cache()

    return info

def plotMINERimage(image, config, info, visualization_channels = None):
    """
    Plot an overview of the MSE over time and the final multiscales.

    image   : The image object
    config  : The configuration dictionary
    info    : Output of the MINER algorithm containing metrics
    """

    im_labels = miner.drawblocks(info['learn_indices_list'],
                                image.shape[:2], config.ksize)
    
    im_labels = im_labels.numpy()

    plt.subplot(1, 2, 1)
    plt.semilogy(info["time_array"], info["mse_array"])
    plt.xlabel('Time (s)')
    plt.ylabel('MSE')
    plt.grid()
    
    im_labels_colored = cm.hsv(im_labels/config.nscales)[..., :3]
    im_labels_colored *= (im_labels[..., np.newaxis] > 0)

    if visualization_channels:
        image = image[:, :, visualization_channels]

    im_marked = image[..., ::-1]*(im_labels[..., np.newaxis] == 0)
    
    im_marked += im_labels_colored
    
    plt.subplot(1, 2, 2)
    plt.imshow(im_marked)
    plt.show()
    
def readAllSatellite(configuration, satellite_part, only_rgb = False):
    root_dir = os.path.join(os.getcwd(), "..", "data", f"Sat_{satellite_part}", "np_downscaled_and_cropped")
    satellite_dataset = SatelliteDataset(root_dir)
    tensors = torch.stack([tensor.transpose(2,0) for tensor in satellite_dataset])
    
    if only_rgb:
        tensors = tensors[..., TRUE_COLOR]
    
    configs = [copy.copy(configuration) for _ in range(len(tensors))]
    
    config_suffices = [f"{index}_p{satellite_part}" for index in range(len(tensors))]

    images = [readTensor(tensor, config_suffices[index], configs[index]) for index, tensor in enumerate(tensors)]
    
    return configs, images