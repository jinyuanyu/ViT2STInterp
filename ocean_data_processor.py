import numpy as np
import os
import glob
import netCDF4 as nc
from torch.utils.data import Dataset, DataLoader
import torch

def load_data_from_directory(directory, scale_factor=1.0):
    file_paths = glob.glob(f'{directory}/*.nc')
    all_data = []
    all_masks = []
    
    for file_path in file_paths:
        dataset = nc.Dataset(file_path, 'r')
        file_name = os.path.basename(file_path)
        prefix = file_name.split('.')[0]
        data_var = dataset.variables[prefix][:]
        data_var = np.expand_dims(data_var, axis=1)
        mask_var = dataset.variables['mask'][:]
        mask_var = np.expand_dims(mask_var, axis=0)
        missing_value = dataset.variables[prefix]._FillValue

        mask = (mask_var == 1).astype(np.float32)
        data_var[data_var == missing_value] = 0
        data_var = np.nan_to_num(data_var, nan=0.0)

        data_var *= scale_factor
        all_data.append(data_var)
        all_masks.append(mask)

        dataset.close()

    all_data = np.concatenate(all_data, axis=1)
    all_masks = np.concatenate(all_masks, axis=0)

    return all_data, all_masks

class OceanDataset(Dataset):
    def __init__(self, data, mask, len_frame=1, use_random_mask=False, mask_ratio=0.0):
        self.data = data
        self.mask = mask
        self.len_frame = len_frame
        self.use_random_mask = use_random_mask
        self.mask_ratio = mask_ratio

    def __len__(self):
        return int(self.data.shape[0] / self.len_frame)

    def __getitem__(self, idx):
        data_sample = self.data[idx*self.len_frame:(idx+1)*self.len_frame, :, :]
        mask_sample = self.mask[0, :, :]

        data_sample = torch.tensor(data_sample, dtype=torch.float32)
        mask_sample = torch.tensor(mask_sample, dtype=torch.float32)

        if self.use_random_mask:
            random_mask = torch.rand_like(mask_sample) < self.mask_ratio
            mask_sample = torch.max(mask_sample, random_mask.float())

        meta_data = (data_sample == 0).float()
        return data_sample, mask_sample, meta_data

def create_dataloader(directory, batch_size=1, len_frame=1, use_random_mask=False, mask_ratio=0.0, shuffle=False, scale_factor=1.0):
    data_var, mask = load_data_from_directory(directory, scale_factor)
    dataset = OceanDataset(data_var, mask, len_frame, use_random_mask, mask_ratio)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
