import os
import torch
import numpy as np
import matplotlib.pyplot as plt

MEANS = np.array([1353.4440, 1115.2527, 1033.7759,  935.1631, 1180.9276, 1966.8143, 2329.8030, 2257.3333,  722.7522,   13.1261, 1781.4470, 1098.1393, 2546.6523]).reshape(1,1,13)
STDS = np.array([63.9742, 150.7762, 183.9667, 272.4519, 223.1591, 348.4093, 445.4806, 519.6234,  96.9965,   1.1972, 370.0003, 296.7262, 491.4442]).reshape(1,1,13)


def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def display_preprocess(image):
    if image.shape[0] == 1:
        image = image[0]

    if len(image.shape) == 2:
        sidelen = int(image.shape[0] ** .5)
        image = image.view(sidelen, sidelen, -1)

    if image.shape[1] == image.shape[2]:
        image = np.transpose(image, (1, 2, 0))

    if image.shape[-1] == 3:
        pass
        # if image.min() < 1e-3:
        #     image = (image / 2) + 0.5
    else:
        # if -100 < image.mean() < 100:
        #     image *= STDS
        #     image += MEANS

        image = image[:, :, [3, 2, 1]]
        # image /= image.max()

    image = (image * 255).numpy().astype(np.uint8)

    return image


def show_image(image: torch.tensor):
    image = display_preprocess(image)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def show_images(img1, img2):
    img1 = display_preprocess(img1)
    img2 = display_preprocess(img2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img1, cmap='gray')
    axes[0].axis('off')

    axes[1].imshow(img2, cmap='gray')
    axes[1].axis('off')

    plt.show()


def psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def postprocess_model_output(image):
    image = image.cpu().detach()

    if image.shape[0] == 1:
        image = image[0]

    sidelen = int(image.shape[0] ** .5)
    image = image.view(sidelen, sidelen, -1)

    if image.shape[-1] == 3:
        image = (image / 2) + 0.5
    else:
        image *= STDS
        image += MEANS

        image /= image.max()

    return image


def bpp(model, image_size=64, model_file="model.pth", delete_model=True):
    torch.save(model.state_dict(), model_file)

    model_size_bytes = os.path.getsize(model_file)
    model_size_bits = model_size_bytes * 8

    if delete_model:
        os.remove(model_file)

    return model_size_bits / (image_size * image_size)
