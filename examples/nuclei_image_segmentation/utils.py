import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt


def convert_cell_to_img(t, padding=16):
    """Converts pytorch tensor into a Pillow Image. The padding will be removed
    from the resulting image"""
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mu = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    output = t.mul(std)
    output.add_(mu)
    img = to_pil_image(output)
    w, h = img.size
    return img.crop((padding, padding, w - padding, h - padding))


def plot_mask_cells(mask_cells, padding=16):
    """Plots cells with their true mask, predicted mask.

    Parameters
    ----------
    mask_cells: list of tuples (`true_mask`, `predicted_mask`, `cell`)
    padding: int (default=16)
        Padding around mask to remove.
    """
    fig, axes = plt.subplots(len(mask_cells), 3, figsize=(12, 10))
    for idx, (axes, mask_cell) in enumerate(zip(axes, mask_cells), 1):
        ax1, ax2, ax3 = axes
        true_mask, predicted_mask, cell = mask_cell
        plot_mask_cell(
                true_mask, predicted_mask, cell,
                'Type {}'.format(idx),
                ax1, ax2, ax3,
                padding=padding)
    fig.tight_layout()
    return fig, axes


def plot_mask_cell(true_mask,
                   predicted_mask,
                   cell,
                   suffix,
                   ax1,
                   ax2,
                   ax3,
                   padding=16):
    """Plots a single cell with a its true mask and predicuted mask"""
    for ax in [ax1, ax2, ax3]:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    ax1.imshow(true_mask[padding:-padding, padding:-padding], cmap='viridis')
    ax1.set_title('True Mask - {}'.format(suffix))
    ax2.imshow(
        predicted_mask[padding:-padding, padding:-padding], cmap='viridis')
    ax2.set_title('Predicted Mask - {}'.format(suffix))
    ax3.imshow(convert_cell_to_img(cell, padding=padding))
    ax3.set_title('Image - {}'.format(suffix))
    return ax1, ax2, ax3


def plot_masks(mask_1, mask_2, mask_3):
    """Plots three masks"""
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(12, 5))
    for ax in [ax1, ax2, ax3]:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    ax1.set_title("Type 1")
    ax1.imshow(mask_1, cmap='viridis')
    ax2.set_title("Type 2")
    ax2.imshow(mask_2, cmap='viridis')
    ax3.set_title("Type 3")
    ax3.imshow(mask_3, cmap='viridis')
    return ax1, ax2, ax3


def plot_cells(cell_1, cell_2, cell_3):
    """Plots three cells"""
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(12, 5))
    for ax in [ax1, ax2, ax3]:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    ax1.set_title("Type 1")
    ax1.imshow(cell_1)
    ax2.set_title("Type 2")
    ax2.imshow(cell_2)
    ax3.set_title("Type 3")
    ax3.imshow(cell_3)
    return ax1, ax2, ax3
