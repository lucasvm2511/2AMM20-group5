import torch

def psnr(img1, img2, max_pixel_value=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))

    return psnr

class TargetPSNRLoss:
    def __init__(self, target):
        self.target = target

    def __call__(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))

        return (self.target - psnr) / self.target

class PSNREvaluationMetric:
    def __init__(self, max_pixel_value=255.0):
        self.max_pixel_value = max_pixel_value

    def __call__(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        psnr = 20 * torch.log10(self.max_pixel_value / torch.sqrt(mse))

        return psnr
