import argparse
import os
import models
import numpy as np
import torch.nn as nn
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from saver import ModelSaver


def plot_filters(filters, save_path, num_cols=6):
    if not len(filters.shape) == 4:
        raise ValueError("Filters ndarray must be 4-dimensional.")
    if not filters.shape[1] == 3:
        raise ValueError("Filters ndarray has shape {}, must have shape (b, c=3, h, w)."
                         .format(filters.shape))
    num_kernels = filters.shape[0]
    num_rows = 1 + num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols, num_rows))

    # Normalize filter values into 0, 1
    filters -= np.min(filters)
    filters /= np.max(filters)

    # Move channels last
    filters = np.transpose(filters, (0, 2, 3, 1))

    for i in range(filters.shape[0]):
        ax1 = fig.add_subplot(num_rows, num_cols, i + 1)
        ax1.imshow(filters[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='imgs/',
                        help='Output directory for generated images.')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to model checkpoint whose filters will be visualized.')
    parser.add_argument('--model', type=str, required=True, choices=('alexnet', 'vgg19', 'resnet50'),
                        help='Model type.')
    args = parser.parse_args()

    model_dict = {
        'alexnet': models.alexnet,
        'resnet50': models.resnet50,
        'vgg19': models.vgg19_bn,
    }
    model_fn = model_dict[args.model]
    model = model_fn(num_classes=10)
    model = nn.DataParallel(model)
    ckpt_info = ModelSaver.load_model(args.ckpt_path, model)

    filter_dict = {
        'alexnet': 'module.features.0',
        'resnet50': 'module.conv1',
        'vgg19': 'module.features.0'
    }

    img_path = os.path.join(args.output_dir, '{}_{}.png'.format(args.model, ckpt_info['epoch']))
    target_layer = filter_dict[args.model]
    for name, module in model.named_modules():
        if name == target_layer:
            plot_filters(module.weight.data.numpy(), img_path)
