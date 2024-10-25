import os
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset
from PIL import Image

from src.utils import get_mgrid


class EuroSATAllBands(Dataset):
    def __init__(self, root_dir, image_size=64, transform=None):
        self._data_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

        self._tif_files = [
            os.path.join(root_dir, directory, img)
            for directory in self._data_folders
            for img in os.listdir(os.path.join(root_dir, directory))
            if img.endswith('.tif')
        ]

        self._coords = get_mgrid(image_size, 2)
        self._transform = transform

    def __len__(self):
        return len(self._tif_files)

    def __getitem__(self, idx):
        with rasterio.open(self._tif_files[idx]) as src:
            image = src.read().astype(np.float32)

        image = torch.tensor(image)

        if self._transform:
            image = self._transform(image)

        return self._coords, image.permute(1, 2, 0).view(-1, 13)


class EuroSATRGB(Dataset):
    def __init__(self, root_dir, image_size=64, transform=None):
        self._data_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

        self._jpg_files = [
            os.path.join(root_dir, directory, img)
            for directory in self._data_folders
            for img in os.listdir(os.path.join(root_dir, directory))
            if img.endswith('.jpg')
        ]

        self._coords = get_mgrid(image_size, 2)
        self._transform = transform

    def __len__(self):
        return len(self._jpg_files)

    def __getitem__(self, idx):
        img_path = self._jpg_files[idx]
        image = Image.open(img_path).convert("RGB")

        if self._transform:
            image = self._transform(image)

        return self._coords, image.permute(1, 2, 0).view(-1, 3)


class NumpyDatasetAllBands(Dataset):
    def __init__(self, root_dir, transform=None):
        self._npy_files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".npy")
        ]
        self._transform = transform

    def __len__(self):
        return len(self._npy_files)

    def __getitem__(self, idx):
        image = np.load(self._npy_files[idx]).astype(np.float32)
        image = np.transpose(image, (1, 2, 0))
        image /= image.max()

        sidelen = image.shape[0]

        if self._transform:
            image = self._transform(image)

        return get_mgrid(sidelen, 2), image.permute(1, 2, 0).view(-1, 13)


class NumpyDatasetRGB(Dataset):
    def __init__(self, root_dir, transform=None):
        self._npy_files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith(".npy")
        ]
        self._transform = transform

    def __len__(self):
        return len(self._npy_files)

    def __getitem__(self, idx):
        image = np.load(self._npy_files[idx]).astype(np.float32)
        image = image[[3,2,1], :, :]
        image = np.transpose(image, (1, 2, 0))
        image /= image.max()

        sidelen = image.shape[0]

        if self._transform:
            image = self._transform(image)

        return get_mgrid(sidelen, 2), image.permute(1, 2, 0).view(-1, 3)