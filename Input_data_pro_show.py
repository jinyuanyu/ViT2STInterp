# ocean_data_processor.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import torch.nn.functional as F

#* seq:(frames,height,width)
def show_sequence(seq,sample=8,title='',channel=1,vmin=0, vmax=255):

    assert (channel in [1, 3]), "Input must have 1 (grayscale) or 3 (RGB) channels"
    
    # Convert tensor to NumPy array if necessary
    if isinstance(seq, torch.Tensor):
        seq = seq.detach().cpu().numpy()

    fig = plt.figure(figsize=(20, 2.5))
    fig.suptitle(title, fontsize=16)
    
    grid = ImageGrid(fig, 111, nrows_ncols=(1, sample), axes_pad=0.1)
    
    for ax, img in zip(grid, seq):
        if channel == 3:  # RGB image
            IMAGENET_STD=np.array([0.225,0.225,0.225])
            IMAGENET_MEAN=np.array([0.45,0.45,0.45])
            img=img*IMAGENET_STD+IMAGENET_MEAN
            # Scale image and convert to uint8 for display
            img = (img * 255).clip(0, 255).astype('uint8')
            ax.imshow(img)
        elif channel == 1:  # Grayscale image
            img = (img * 10000).clip(0, 255).astype('uint8')
            ax.imshow(img, vmin=vmin, vmax=vmax)
        
        ax.set_axis_off()
    
    plt.show()
    return

def resize_tensor(tensor, target_size, mode='bilinear', align_corners=False):
    """
    Resize a 5D tensor to the target size using padding and interpolation.

    Parameters:
        tensor (torch.Tensor): Tensor of shape (batch_size, frames, channels, height, width).
        target_size (tuple): Target size (new_height, new_width).
        mode (str): Interpolation mode to calculate output values ('bilinear', 'nearest', etc.).
        align_corners (bool): If True, the corner pixels of the input and output tensors are aligned.
    
    Returns:
        torch.Tensor: Resized tensor.
    """
    _, _, _, current_height, current_width = tensor.shape
    target_height, target_width = target_size

    # Calculate padding
    pad_height = max(0, target_height - current_height)
    pad_width = max(0, target_width - current_width)

    # Calculate padding for each side
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Pad the tensor
    padded_tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))

    return padded_tensor
