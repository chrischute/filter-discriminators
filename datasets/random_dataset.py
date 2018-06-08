import h5py
import numpy as np
import torch
import torch.utils.data as data


class RandomDataset(data.Dataset):
    def __init__(self, hdf5_path, is_training):
        self.is_training = is_training
        self.phase = 'train' if self.is_training else 'test'
        with h5py.File(hdf5_path, 'r') as hdf5_fh:
            self.dataset = hdf5_fh[self.phase][...]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        image_tensor = torch.from_numpy(img.astype(np.float32))
        label_tensor = torch.tensor(idx % 10, dtype=torch.int64)

        return image_tensor, label_tensor
