import copy
import torch

import yaml
import torch
import sys

from torch.utils.tensorboard import SummaryWriter
from input_encoding import generate_grid
from models.nif import NIF
from phases.fitting import fit_with_config

from utils import dump_model_stats, load_configuration, load_device
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from PIL import Image

import numpy as np


def main():
    torch.random.manual_seed(1337)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)

    config_path = sys.argv[1]
    file_path = sys.argv[2]
    model_dump_path = sys.argv[3]

    fit(config_path, file_path, model_dump_path)

def normalize_tensor(tens:torch.Tensor, mean:float=0.5, std:float=0.5) -> torch.Tensor:
    max_val = tens.max()
    min_val = tens.min()
    return (tens-min_val)/(max_val-min_val)

def fit(config_path, file_path, compressed_path, model_dump_path=None, rgb=False):
    print("Loading configuration...")
    device = load_device()

    config = load_configuration(config_path)

    writer = SummaryWriter(log_dir = f"runs/{config_path}_{file_path}_fitting")

    writer.add_text("config", "```\n" + str(config).replace('\n', '\n\n') + "\n```")

    print("Loading images...")
    
    # This was added by 2AMM20 NF Group 5 -------------------------------------------
    # Check if the file at file_path is a .png or .npy file
    image = Image.new('RGB', (64, 64)) # initialise an empty 64*64 image

    # preprocess a png image
    if file_path[-4:] == ".png":
        image = Image.open(file_path)
        (height, width) = (image.size[1], image.size[0])
        transform = Compose([
            Resize((height, width)),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])
        image_tensor = transform(image)
    
    # preprocess a npy numpy.ndarray    
    elif file_path[-4:] == ".npy":
        npy_array = np.load(file_path)
        # image_t = torch.from_numpy(npy_array.astype('uint8'))
        image_t = torch.from_numpy(npy_array.astype(np.float32))

        # Transpose the image without touching the channels
        (channels, height, width) = (image_t.shape[0], image_t.shape[2], image_t.shape[1]) 
        image_resized = image_t.view((channels, height, width))

        # Normalize the image
        image_tensor = normalize_tensor(image_resized)
        
        # if working with an rgb image, only select the relevant dimensions to do so
        #print('rgb = ', rgb)
        if rgb:
          image_tensor = image_tensor[[3,2,1]]
          
    # -------------------------------------------------------------------------------

    image_tensor = image_tensor.to(device)

    print("Generating grid...")
    grid = generate_grid(config, width, height, device)
    
    print("Loading model...")
    params = config["model"]
    params["input_features"] = grid.size(-1)
    model = NIF(**params, rgb=rgb, device=device, writer=writer).to(device)
    model.initialize_weights()

    print(model)
    dump_model_stats(model, width, height, writer)

    final_psnr = fit_with_config(config["fitting"], model, grid, image_tensor, compressed_path, 
                                             verbose=True, writer=writer)
    print(f"Final PSNR: {final_psnr}")

    final_state_dict = copy.deepcopy(model.state_dict())
    final_state_dict["__meta"] = {
        "width": width,
        "height": height
    }

    if model_dump_path:
        print("Model weights dump...")
        model.eval()
        torch.save(final_state_dict, model_dump_path)
        
    #print('final_state_dict = ')
    #print(final_state_dict)
    #print()

    return final_state_dict

if __name__ == "__main__":
    main()
