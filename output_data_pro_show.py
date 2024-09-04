import numpy as np
from einops import rearrange

def rearange_output(input_tensor, reconstructed_patches, bool_masked_pos, config):
    patch_size = (config.patch_size, config.patch_size)

    # Permute and convert to NumPy
    orig_frames = input_tensor.permute(0, 1, 3, 4, 2).cpu().numpy()

    # Squeeze and normalize patches
    img_squeeze = rearrange(
        orig_frames, 
        'b (t p0) (h p1) (w p2) c -> b (t h w) (p0 p1 p2) c', 
        p0=config.tubelet_size, p1=patch_size[0], p2=patch_size[1]
    )
    img_mean = np.mean(img_squeeze, axis=-2, keepdims=True)
    img_variance = np.var(img_squeeze, axis=-2, ddof=1, keepdims=True)
    img_norm = (img_squeeze - img_mean) / (np.sqrt(img_variance) + 1e-6)

    # Prepare patches
    img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
    img_patch[bool_masked_pos.cpu()] = reconstructed_patches.cpu().detach().numpy()

    # Reconstruct the image
    rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=config.num_channels)
    img_mean = np.mean(img_squeeze, axis=-2, keepdims=True)
    img_std = np.sqrt(np.var(img_squeeze, axis=-2, ddof=1, keepdims=True) + 1e-6) 
    rec_img = rec_img * img_std + img_mean
    rec_img = rearrange(
        rec_img, 
        'b (t h w) (p0 p1 p2) c -> b (t p0) (h p1) (w p2) c', 
        p0=config.tubelet_size, p1=patch_size[0], p2=patch_size[1], 
        h=config.image_size // config.patch_size, w=config.image_size // config.patch_size
    )

    # Prepare mask
    mask = np.ones_like(img_patch)
    mask[bool_masked_pos.cpu()] = 0
    mask = rearrange(mask, 'b n (p c) -> b n p c', c=config.num_channels)
    mask = rearrange(
        mask,
        'b (t h w) (p0 pl p2) c -> b (t p0) (h pl) (w p2) c',
        p0=config.tubelet_size, pl=patch_size[0], p2=patch_size[1],
        h=config.image_size // config.patch_size, w=config.image_size // config.patch_size
    )

    return rec_img, mask
