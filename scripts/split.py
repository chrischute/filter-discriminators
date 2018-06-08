import argparse
import json
import models
import os
import random
import torch.nn as nn

from saver import ModelSaver
from sklearn.model_selection import train_test_split
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
    parser.add_argument('--meta_path', type=str, default='filters/meta.json',
                        help='Path to metadata file.')
    parser.add_argument('--train_size', type=float, default=0.9,
                        help='Ratio of training set size to total dataset size.')
    args = parser.parse_args()

    # Load metadata for any filters that have already been extracted
    if os.path.exists(args.meta_path):
        print('Adding to existing metadata file...')
        with open(args.meta_path, 'r') as meta_fh:
            meta_dict = json.load(meta_fh)
    else:
        raise ValueError('Invalid metadata file: {}'.format(args.meta_path))

    resnet_ckpts = [c for c in meta_dict if c.startswith('r')]
    vgg_ckpts = [c for c in meta_dict if c.startswith('v')]
    alexnet_ckpts = [c for c in meta_dict if c.startswith('a')]
    all_ckpts = resnet_ckpts + vgg_ckpts + alexnet_ckpts
    quartiles = [int(meta_dict[c]['tp_score'] * 4 // 1) for c in all_ckpts]
    print(quartiles)
    train_ckpts, test_ckpts = train_test_split(all_ckpts, test_size=0.25, stratify=quartiles)

    for ckpt_name, ckpt_dict in meta_dict.items():
        if ckpt_name in test_ckpts:
            ckpt_dict['phase'] = 'val'
        else:
            ckpt_dict['phase'] = 'train'

    print('Total set size: {}'.format(len(meta_dict)))
    print('Train set size: {}'.format(sum([1 if d['phase'] == 'train' else 0 for d in meta_dict.values()])))
    print('Val set size: {}'.format(sum([1 if d['phase'] == 'val' else 0 for d in meta_dict.values()])))
    print('Bottom quartile val set size: {}'.format(sum([1 if (d['phase'] == 'val' and d['tp_score'] <= 0.25) else 0 for d in meta_dict.values()])))

    # Dump scores to JSON
    with open(args.meta_path, 'w') as meta_fh:
        print('Dumping metadata file to {}...'.format(args.meta_path))
        json.dump(meta_dict, meta_fh, indent=4, sort_keys=True)
        meta_fh.write('\n')
