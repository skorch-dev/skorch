#!/usr/bin/env python
"""Each sample contains multiple masks. This script combines theese masks into
one image, thus creating one mask for each sample.
"""
from multiprocessing import Pool
from pathlib import Path
from contextlib import ExitStack

from tqdm import tqdm
from PIL import Image


def combine_masks(mask_root_dir):
    mask_output_fn = mask_root_dir / 'mask.png'
    if mask_output_fn.exists():
        return
    mask_fn_iter = mask_root_dir.glob('masks/*.png')
    img = Image.open(next(mask_fn_iter))
    for fn in mask_fn_iter:
        mask = Image.open(fn)
        img.paste(mask, (0, 0), mask)
    img.save(mask_output_fn)


# Combine masks into one
samples_dirs = list(d for d in Path('data/cells').iterdir() if d.is_dir())
with ExitStack() as stack:
    pool = stack.enter_context(Pool())
    pbar = stack.enter_context(tqdm(total=len(samples_dirs)))
    for _ in tqdm(pool.imap_unordered(combine_masks, samples_dirs)):
        pbar.update()
