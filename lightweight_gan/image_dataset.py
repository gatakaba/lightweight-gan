from functools import partial
from pathlib import Path
from random import random

import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

from lightweight_gan.lightweight_gan import EXTS


def exists(val):
    return val is not None


def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image


def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f"image with invalid number of channels given {channels}")

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))


class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)


class ImageDataset(Dataset):
    def __init__(self, folder, image_size, transparent=False, aug_prob=0.0):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f"{folder}").glob(f"**/*.{ext}")]
        assert len(self.paths) > 0, f"No images were found in {folder} for training"

        convert_image_fn = partial(convert_image_to, "RGBA" if transparent else "RGB")
        # num_channels = 3 if not transparent else 4

        self.transform = transforms.Compose(
            [
                transforms.Lambda(convert_image_fn),
                transforms.Lambda(partial(resize_to_minimum_size, image_size)),
                transforms.Resize(image_size),
                RandomApply(
                    aug_prob,
                    transforms.RandomResizedCrop(
                        image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)
                    ),
                    transforms.CenterCrop(image_size),
                ),
                transforms.ToTensor(),
                transforms.Lambda(expand_greyscale(transparent)),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)
