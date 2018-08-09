"""Dataset for cells"""
from itertools import product, chain
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import (pad, to_tensor, normalize,
                                               hflip, vflip, crop)
from PIL import Image

def calcuate_bboxes(im_shape, patch_size):
    """Calculate bound boxes based on image shape and size of the bounding box
    given by `patch_size`"""
    h, w = im_shape
    ph, pw = patch_size

    steps_h = chain(range(0, h - ph, ph), [h - ph])
    steps_w = chain(range(0, w - pw, pw), [w - pw])

    return product(steps_h, steps_w)

class PatchedDataset(Dataset):
    """Creates patches of cells.

    Parameters
    ----------
    base_dataset: CellsDataset
        Dataset of cells
    patch_size: tuple of ints (default=(256, 256))
        The size of each patch
    random_flips: bool (default=False)
        If true, patches and masks will be randomly flipped horizontally and
        vertically.
    padding: int (default=16)
        Amount of paddding around each image and mask
    """
    def __init__(self,
                 base_dataset,
                 patch_size=(256, 256),
                 random_flips=False,
                 padding=16):
        super().__init__()
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.random_flips = random_flips

        image_cache = []
        mask_cache = []
        h, w = patch_size[0] + 2 * padding, patch_size[1] + 2 * padding
        for idx, (cell, mask) in enumerate(self.base_dataset):
            bboxes = calcuate_bboxes((h, w), self.patch_size)

            cell = pad(cell, padding, padding_mode='reflect')
            mask = pad(mask, padding, padding_mode='reflect')

            for i, j in bboxes:
                image_cache.append(np.array(crop(cell, i, j, h, w)))
                mask_cache.append(np.array(crop(mask, i, j, h, w)))

        self.image_cache = np.stack(image_cache)
        self.mask_cache = np.stack(mask_cache)

    def __len__(self):
        return self.mask_cache.shape[0]

    def __getitem__(self, idx):
        cell = self.image_cache[idx]
        mask = self.mask_cache[idx]

        cell, mask = Image.fromarray(cell), Image.fromarray(mask)

        if self.random_flips:
            if random.random() < 0.5:
                cell = hflip(cell)
                mask = hflip(mask)

            if random.random() < 0.5:
                cell = vflip(cell)
                mask = vflip(mask)

        cell = to_tensor(cell)
        mask = torch.as_tensor((np.array(mask) == 255).astype('float32'))

        # mean and std of imagenet
        cell = normalize(cell, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return cell, mask


class CellsDataset(Dataset):
    """Constructs cell dataset"""
    def __init__(self, sample_dirs):
        super().__init__()
        self.sample_dirs = sample_dirs

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        cell_fn = (sample_dir / 'images' / sample_dir.name).with_suffix('.png')
        mask_fn = sample_dir / 'mask.png'

        cell, mask = Image.open(cell_fn).convert('RGB'), Image.open(mask_fn)
        assert cell.size == mask.size
        return cell, mask
