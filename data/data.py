import torch
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import ColorJitter, RandomApply, GaussianBlur
from data.LOLdataset import *
from data.eval_sets import *
from data.SICE_blur_SID import *
from data.fivek import *

class AddGaussianNoise:
    def __init__(self, std=0.01, p=0.5):
        self.std = float(std)
        self.p = float(p)

    def __call__(self, tensor):
        if self.p <= 0:
            return tensor
        if torch.rand(1).item() >= self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


def transform1(size=256, aug_color=True, aug_blur=False, aug_noise=False, noise_std=0.01):
    augments = []
    if aug_color:
        augments.append(RandomApply([ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.02)], p=0.8))
    if aug_blur:
        augments.append(RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2))
    return Compose([
        RandomCrop((size, size)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        *augments,
        ToTensor(),
        AddGaussianNoise(std=noise_std, p=0.5) if aug_noise else lambda x: x,
    ])

def transform2():
    return Compose([ToTensor()])



def get_lol_training_set(data_dir,size, aug_color=True, aug_blur=False, aug_noise=False, noise_std=0.01):
    return LOLDatasetFromFolder(data_dir, transform=transform1(size, aug_color, aug_blur, aug_noise, noise_std))


def get_lol_v2_training_set(data_dir,size, aug_color=True, aug_blur=False, aug_noise=False, noise_std=0.01):
    return LOLv2DatasetFromFolder(data_dir, transform=transform1(size, aug_color, aug_blur, aug_noise, noise_std))


def get_training_set_blur(data_dir,size, aug_color=True, aug_blur=False, aug_noise=False, noise_std=0.01):
    return LOLBlurDatasetFromFolder(data_dir, transform=transform1(size, aug_color, aug_blur, aug_noise, noise_std))


def get_lol_v2_syn_training_set(data_dir,size, aug_color=True, aug_blur=False, aug_noise=False, noise_std=0.01):
    return LOLv2SynDatasetFromFolder(data_dir, transform=transform1(size, aug_color, aug_blur, aug_noise, noise_std))


def get_SID_training_set(data_dir,size, aug_color=True, aug_blur=False, aug_noise=False, noise_std=0.01):
    return SIDDatasetFromFolder(data_dir, transform=transform1(size, aug_color, aug_blur, aug_noise, noise_std))


def get_SICE_training_set(data_dir,size, aug_color=True, aug_blur=False, aug_noise=False, noise_std=0.01):
    return SICEDatasetFromFolder(data_dir, transform=transform1(size, aug_color, aug_blur, aug_noise, noise_std))

def get_SICE_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())

def get_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())

def get_fivek_training_set(data_dir,size, aug_color=True, aug_blur=False, aug_noise=False, noise_std=0.01):
    return FiveKDatasetFromFolder(data_dir, transform=transform1(size, aug_color, aug_blur, aug_noise, noise_std))

def get_fivek_eval_set(data_dir):
    return SICEDatasetFromFolderEval(data_dir, transform=transform2())

def get_mydata_training_set(data_dir, size, aug_color=True, aug_blur=False, aug_noise=False, noise_std=0.01):
    return LOLDatasetFromFolder(data_dir, transform=transform1(size, aug_color, aug_blur, aug_noise, noise_std))

def get_mydata_eval_set(data_dir):
    return DatasetFromFolderEval(data_dir, transform=transform2())