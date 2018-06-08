import numpy as np
import json
import os
import torch

from torch.utils.data.dataset import Dataset
from .filter_datum import FilterDatum


class FilterDataset(Dataset):
    def __init__(self, model_name, data_dir, is_training):
        self.data_dir = data_dir
        self.is_training = is_training
        with open(os.path.join(data_dir, 'meta.json'), 'r') as meta_fh:
            meta_dict = json.load(meta_fh)

        self.filters = []
        for ckpt_name, ckpt_dict in meta_dict.items():
            if ckpt_name.startswith(model_name[0]):
                if (ckpt_dict['phase'] == ('train' if self.is_training else 'val')
                   and 0. <= ckpt_dict['tp_score'] <= 1.):
                    filters_np = np.load(os.path.join(self.data_dir, ckpt_name + '.npy'))
                    self.filters += [FilterDatum(f, ckpt_dict['tp_score']) for f in filters_np]

        print('Loaded {} filters...'.format(len(self.filters)))

    def __getitem__(self, idx):
        """Get a filter and associated TP score."""
        filter_datum = self.filters[idx]

        filter_tensor = torch.from_numpy(filter_datum.filter_np)
        tp_score_tensor = torch.tensor([filter_datum.tp_score], dtype=torch.float32)

        return filter_tensor, tp_score_tensor

    def __len__(self):
        return len(self.filters)
