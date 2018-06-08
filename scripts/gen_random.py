"""Generate a Dataset that looks like CIFAR10, but whose images are random noise."""
import argparse
import h5py
import numpy as np
import os

NUM_TRAIN = 50000
NUM_TEST = 10000
IMG_SIZE = 32
NUM_CHANNELS = 3


def gen_random(output_dir):
    """Generate a random dataset in an HDF5 file."""
    with h5py.File(os.path.join(output_dir, 'random.hdf5'), 'w') as hdf5_fh:
        # Note: We don't store labels. Image at index idx has label (idx % 10).

        train_set = (np.random.rand(NUM_TRAIN, NUM_CHANNELS, IMG_SIZE, IMG_SIZE) * 255).astype(np.uint8)
        hdf5_fh.create_dataset('train', data=train_set)

        test_set = (np.random.rand(NUM_TEST, NUM_CHANNELS, IMG_SIZE, IMG_SIZE) * 255).astype(np.uint8)
        hdf5_fh.create_dataset('test', data=test_set)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for random dataset.')
    args = parser.parse_args()

    gen_random(args.output_dir)
