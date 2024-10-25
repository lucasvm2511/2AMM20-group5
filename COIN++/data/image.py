import imageio
import requests
import torch
import torchvision
import cv2
import zipfile
import os
import kaggle
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
import rasterio
import tifffile as tiff


class CIFAR10(torchvision.datasets.CIFAR10):
    """CIFAR10 dataset without labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.data[index])
        else:
            return self.data[index]


class MNIST(torchvision.datasets.MNIST):
    """MNIST dataset without labels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        # Add channel dimension, convert to float and normalize to [0, 1]
        datapoint = self.data[index].unsqueeze(0).float() / 255.0
        print(datapoint.shape)
        if self.transform:
            return self.transform(datapoint)
        else:
            return datapoint
        
class EuroSatDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, download_path="data/EuroSat", transform=None):
        self.download_path = download_path
        self.df_path = df_path
        self.transform = transform

        # Download dataset if it does not exist
        if not os.path.exists(self.download_path):
            self.download_dataset(self.download_path)

        if not os.path.exists(self.df_path):
            raise FileNotFoundError(f"Data file {self.df_path} does not exist.")

        # Load the dataset information
        self.df = pd.read_csv(self.df_path)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "Filename"]
        image = cv2.imread(os.getcwd()+'/data/EuroSat/EuroSAT/'+path)

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.df)

    @staticmethod
    def download_dataset(download_path="data"):
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        
        # Set up Kaggle API credentials (ensure you have kaggle.json in ~/.kaggle or provide the environment variable)
        os.environ['KAGGLE_USERNAME'] = 'lucasvm2511'
        os.environ['KAGGLE_KEY'] = '32f1ad09a5171f1c00883bef693554e1'

        # Download the dataset
        kaggle.api.dataset_download_files("apollo2506/eurosat-dataset", path=download_path, unzip=True)

        # Extract if necessary
        dataset_zip_path = os.path.join(download_path, "eurosat-dataset.zip")
        if os.path.exists(dataset_zip_path):
            with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            os.remove(dataset_zip_path)

        print("Dataset downloaded and extracted successfully.")

class EuroSatallDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, download_path="data/EuroSat", transform=None):
        self.download_path = download_path
        self.df_path = df_path
        self.transform = transform

        # Download dataset if it does not exist
        if not os.path.exists(self.download_path):
            self.download_dataset(self.download_path)

        if not os.path.exists(self.df_path):
            raise FileNotFoundError(f"Data file {self.df_path} does not exist.")

        # Load the dataset information
        self.df = pd.read_csv(self.df_path)

    def __getitem__(self, idx):
        # print(self.df.head)
        path = self.df.loc[idx, "Filename"]
        image = tiff.imread(os.getcwd()+'/data/EuroSat/EuroSATallBands/'+path)
        image_normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        image_uint8 = image_normalized.astype(np.uint8)
        image = image_uint8

        if self.transform:
            image = self.transform(image)
        

        return image

    def __len__(self):
        return len(self.df)

    @staticmethod
    def download_dataset(download_path="data"):
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        
        # Set up Kaggle API credentials (ensure you have kaggle.json in ~/.kaggle or provide the environment variable)
        os.environ['KAGGLE_USERNAME'] = 'lucasvm2511'
        os.environ['KAGGLE_KEY'] = '32f1ad09a5171f1c00883bef693554e1'

        # Download the dataset
        kaggle.api.dataset_download_files("apollo2506/eurosat-dataset", path=download_path, unzip=True)

        # Extract if necessary
        dataset_zip_path = os.path.join(download_path, "eurosat-dataset.zip")
        if os.path.exists(dataset_zip_path):
            with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            os.remove(dataset_zip_path)

        print("Dataset downloaded and extracted successfully.")

class LargeDataset(torch.utils.data.Dataset):
    def __init__(self, df_path, download_path="data/np_downscaled_and_cropped", transform=None):
        self.download_path = download_path
        self.df_path = df_path
        self.transform = transform

        if not os.path.exists(self.df_path):
            raise FileNotFoundError(f"Data file {self.df_path} does not exist.")

        # Load the dataset information
        self.df = pd.read_csv(self.df_path, delimiter=';')

    def __getitem__(self, idx):
        path = self.df['Filename'].iloc[idx]

        q = np.load(os.getcwd()+'/data/np_downscaled_and_cropped/'+path)
        image_transposed = q[:,:,:]
        image_transposed = np.transpose(image_transposed, (1, 2, 0))

        # Min-max normalization
        image_normalized = image_transposed.astype(np.float32)
        image_normalized = image_normalized/image_normalized.max()

        # Uncomment to use only RGB bands
        # image_normalized = image_normalized[:,:,1:4]

        # image_normalized = cv2.normalize(image_transposed, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # image = image_normalized.astype(np.uint8)

        image = image_normalized

        if self.transform:
            image = self.transform(image)
        

        return image

    def __len__(self):
        return len(self.df)



class Kodak(torch.utils.data.Dataset):
    """Kodak dataset."""

    base_url = "http://r0k.us/graphics/kodak/kodak/"
    num_images = 24
    width = 768
    height = 512
    resolution_hw = (height, width)

    def __init__(
        self,
        root: Path = Path.cwd() / "kodak-dataset",
        transform: Optional[Callable] = None,
        download: bool = False,
    ):
        self.root = root

        self.transform = transform

        if download:
            self.download()

        self.data = tuple(
            imageio.imread(self.root / f"kodim{idx + 1:02}.png")
            for idx in range(self.num_images)
        )

    def _check_exists(self) -> bool:
        # This can be obviously be improved for instance by comparing checksums.
        return (
            self.root.exists() and len(list(self.root.glob("*.png"))) == self.num_images
        )

    def download(self):
        if self._check_exists():
            return

        self.root.mkdir(parents=True, exist_ok=True)

        print(f"Downloading Kodak dataset to {self.root}...")

        for idx in range(self.num_images):
            filename = f"kodim{idx + 1:02}.png"
            with open(self.root / filename, "wb") as f:
                f.write(
                    requests.get(
                        f"http://r0k.us/graphics/kodak/kodak/{filename}"
                    ).content
                )

        print("Done!")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        image = self.data[index]
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __repr__(self) -> str:
        head = "Dataset Kodak"
        body = []
        body.append(f"Number of images: {self.num_images}")
        body.append(f"Root location: {self.root}")
        lines = [head] + ["    " + line for line in body]
        return "\n".join(lines)
