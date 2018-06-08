import argparse
import json
import models
import numpy as np
import os
import random
import torch.nn as nn
import util

from saver import ModelSaver
from tqdm import tqdm


def extract_filters(ckpt_path, model_name):
    """Get numpy array, val_loss from a saved checkpoint."""
    model_dict = {
        'alexnet': models.alexnet,
        'resnet50': models.resnet50,
        'vgg19': models.vgg19_bn,
    }
    model_fn = model_dict[model_name]
    model = model_fn(num_classes=10)
    model = nn.DataParallel(model)
    ckpt_info = ModelSaver.load_model(ckpt_path, model)

    filter_dict = {
        'alexnet': 'module.features.0',
        'resnet50': 'module.conv1',
        'vgg19': 'module.features.0'
    }
    target_layer = filter_dict[model_name]
    filters_np = None
    for name, module in model.named_modules():
        if name == target_layer:
            filters_np = module.weight.data.cpu().numpy()
            break

    if filters_np is None:
        raise RuntimeError('Could not find filters for layer {}'.format(target_layer))

    return filters_np, ckpt_info['epoch'], ckpt_info['val_loss']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='filters/',
                        help='Output directory for extracted filters.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to directory of checkpoints whose filters will be extracted.')
    parser.add_argument('--model', type=str, required=True, choices=('alexnet', 'vgg19', 'resnet50'),
                        help='Model type.')
    args = parser.parse_args()

    # Load metadata for any filters that have already been extracted
    meta_path = os.path.join(args.output_dir, 'meta.json')
    if os.path.exists(meta_path):
        print('Adding to existing metadata file...')
        with open(meta_path, 'r') as meta_fh:
            meta_dict = json.load(meta_fh)
    else:
        print('Creating new metadata file...')
        meta_dict = {}

    # Extract filters from checkpoints in input_dir
    model_id = args.model[0] + '_' + ''.join([random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(5)])
    ckpt_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith('.pth.tar')]
    best_val_loss = float('inf')
    print('Extracting filters from {} ckpts...'.format(len(ckpt_paths)))
    for ckpt_path_ in tqdm(ckpt_paths):
        filters, epoch, val_loss = extract_filters(ckpt_path_, args.model)
        ckpt_name = model_id + '_{}'.format(epoch)
        meta_dict[ckpt_name] = {
            'val_loss': val_loss
        }
        best_val_loss = min(val_loss, best_val_loss)
        np.save(os.path.join(args.output_dir, ckpt_name + '.npy'), filters)

    # Assign a TP score to each checkpoint
    for ckpt_dict in meta_dict.values():
        ckpt_dict['tp_score'] = util.get_tp_score(ckpt_dict['val_loss'], best_val_loss, num_classes=10)

    # Dump scores to JSON
    with open(meta_path, 'w') as meta_fh:
        print('Dumping metadata file to {}...'.format(meta_path))
        json.dump(meta_dict, meta_fh, indent=4, sort_keys=True)
        meta_fh.write('\n')
