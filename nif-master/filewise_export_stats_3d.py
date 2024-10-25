import os
import json
import torch
import sys
import numpy as np

from skimage import io
from losses.psnr import psnr

from pytorch_msssim import ms_ssim, ssim
from serialization import deserialize_state_dict

from utils import calculate_state_dict_size, load_device
from fit import normalize_tensor

def ms_ssim_reshape(tensor):
    return tensor.movedim(-1, 0).unsqueeze(0)

def main():
    print("Loading device...")
    device = load_device(True)

    print("Loading parameters...")
    original_file_path = sys.argv[1]
    reconstructed_file_path = sys.argv[2]
    stats_path = sys.argv[3]
    compressed_file_path = sys.argv[4]

    print("Calculating compressed state size...")
    # extract filesize in bytes (1 byte = 8 bits)
    compressed_file_size = os.stat(compressed_file_path).st_size

    print("Loading images...")
    # original_image_tensor = torch.from_numpy(io.imread(original_file_path)).to(device).to(torch.float32)
    # reconstructed_image_tensor = torch.from_numpy(io.imread(reconstructed_file_path)).to(device).to(torch.float32)
    
    # Load the original file, which is a npy file
    original_image_tensor = torch.from_numpy(np.load(original_file_path).astype('float32')).to(device).to(torch.float32) 
    # Load in the reconstruced file, which is a npy file
    reconstructed_image_tensor = torch.from_numpy(np.load(reconstructed_file_path).astype('uint8')).to(device).to(torch.float32) 

    # Normalize the image tensors
    original_image_tensor = normalize_tensor(original_image_tensor)
    reconstructed_image_tensor = normalize_tensor(reconstructed_image_tensor)
    
    # since we are working with an rgb image here, we will now have to select the corresponding dimensions
    original_image_tensor = original_image_tensor[[3,2,1]]

    pixels = original_image_tensor.nelement() / 3.0

    print("Calculating stats...")

    # try:
    #     ssim_value = ssim(ms_ssim_reshape(original_image_tensor), ms_ssim_reshape(reconstructed_image_tensor)).item()
    # except Exception as e:
    #     print(f"Cannot calculate SSIM: {e}")
    #     ssim_value = None

    # try:
    #     ms_ssim_value = ms_ssim(ms_ssim_reshape(original_image_tensor), ms_ssim_reshape(reconstructed_image_tensor)).item()
    # except Exception as e:
    #     print(f"Cannot calculate MS-SSIM: {e}")
    #     ms_ssim_value = None

    try:
        compressed_state_dict = deserialize_state_dict(compressed_file_path)
    except Exception as e:
        print(f"WARNING: Cannot deserialize compressed state dict: {e}")
        compressed_state_dict = None

    try:
        if compressed_state_dict is None:
            compressed_state_dict = torch.load(compressed_file_path)

        compressed_state_size = calculate_state_dict_size(compressed_state_dict)
    except Exception as e:
        print(f"WARNING: Cannot calculate state-only bpp: {e}")
        compressed_state_size = 0 

    stats = {
        "psnr": psnr(original_image_tensor, reconstructed_image_tensor, max_pixel_value=1.0).item(), # both images have been normalized to range [0,1]
        # "ms-ssim": ms_ssim_value,
        # "ssim": ssim_value,
        "bpp": (compressed_file_size * 8) / pixels, # this is assuming 8 bit pixels? And you did not account for 13d instead of 3d?
        "state_bpp": (compressed_state_size * 8) / pixels
    }

    try:
        stats["state_size"] = calculate_state_dict_size(torch.load(compressed_file_path))
    except Exception as e:
        pass

    print(stats)

    json.dump(stats, open(stats_path, "w"))

if __name__ == "__main__":
    main()

