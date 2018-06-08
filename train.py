import math
import models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from args import ArgParser
from datasets import FilterDataset, RandomDataset
from logger import TrainLogger
from saver import ModelSaver
from tqdm import tqdm


def train_classifier(args, model):
    """Train a classifier and save its first-layer weights.

    Args:
        args: Command line arguments.
        model: Classifier model to train.
    """
    # Set up data loader
    train_loader, test_loader, classes = get_data_loaders(args.dataset, args.batch_size, args.num_workers)

    # Set up model
    model = nn.DataParallel(model, args.gpu_ids)
    model = model.to(args.device)

    fd = None
    if args.use_fd:
        fd = models.filter_discriminator()
        fd = nn.DataParallel(fd, args.gpu_ids)
        fd = fd.to(args.device)

    # Set up optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                          momentum=args.sgd_momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_gamma)
    if args.model == 'fd':
        post_process = nn.Sigmoid()
        loss_fn = nn.MSELoss().to(args.device)
    else:
        post_process = nn.Sequential()  # Identity
        loss_fn = nn.CrossEntropyLoss().to(args.device)

    # Set up checkpoint saver
    saver = ModelSaver(model, optimizer, scheduler, args.save_dir, {'model': args.model},
                       max_to_keep=args.max_ckpts, device=args.device)

    # Train
    logger = TrainLogger(args, len(train_loader.dataset))
    if args.save_all:
        # Save initialized model weights with validation loss as random
        saver.save(0, math.log(args.num_classes))
    while not logger.is_finished_training():
        logger.start_epoch()

        # Train for one epoch
        model.train()
        fd_lambda = get_fd_lambda(args, logger.epoch)
        for inputs, labels in train_loader:
            logger.start_iter()

            with torch.set_grad_enabled(True):
                # Forward
                outputs = model.forward(inputs.to(args.device))
                outputs = post_process(outputs)
                loss = loss_fn(outputs, labels.to(args.device))
                loss_item = loss.item()

                fd_loss = torch.zeros([], dtype=torch.float32, device='cuda' if args.gpu_ids else 'cpu')
                tp_total = torch.zeros([], dtype=torch.float32, device='cuda' if args.gpu_ids else 'cpu')
                if fd is not None:
                    # Forward FD
                    filters = get_layer_weights(model, filter_dict[args.model])
                    for i in range(0, filters.size(0), args.fd_batch_size):
                        fd_batch = filters[i: i + args.fd_batch_size]
                        tp_scores = F.sigmoid(fd.forward(fd_batch))
                        tp_total += tp_scores.sum()
                    fd_loss = 1. - tp_total / filters.size(0)

                fd_loss_item = fd_loss.item()
                loss += fd_lambda * fd_loss

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logger.end_iter({'std_loss': loss_item, 'fd_loss': fd_loss_item, 'loss': loss_item + fd_loss_item})

        # Evaluate on validation set
        val_loss = evaluate(model, post_process, test_loader, loss_fn, device=args.device)
        logger.write('[epoch {}]: val_loss: {:.3g}'.format(logger.epoch, val_loss))
        logger.write_summaries({'loss': val_loss}, phase='val')
        if args.save_all or logger.epoch in args.save_epochs:
            saver.save(logger.epoch, val_loss)

        logger.end_epoch()
        scheduler.step()


def get_fd_lambda(args, epoch):
    """Get filter discriminator loss multiplier for a given epoch."""
    fd_lambda = args.fd_lambda
    if args.fixed_lambda:
        return fd_lambda
    else:
        # Decay as 2^(-epoch)
        return 2 ** -epoch * args.fd_lambda


def get_layer_weights(model, layer_name):
    """Get reference to weights of a layer."""
    for name, layer in model.named_modules():
        if name == layer_name:
            return layer.weight
    raise RuntimeError('Cannot find layer with name {}'.format(layer_name))


def evaluate(model, post_process, data_loader, loss_fn, device='cpu'):
    """Evaluate the model."""
    model.eval()
    losses = []
    print('Evaluating model...')
    for inputs, labels in tqdm(data_loader):
        with torch.no_grad():
            # Forward
            outputs = model(inputs.to(device))
            outputs = post_process(outputs)
            loss = loss_fn(outputs, labels.to(device))
            losses.append(loss.item())

    return np.mean(losses)


def get_data_loaders(dataset, batch_size, num_workers):
    """Get the `DataLoader`s for this experiment.

    Args:
        dataset: Name of dataset to load.
        batch_size: Batch size for each `DataLoader`.
        num_workers: Number of worker threads for each `DataLoader`.

    Returns:
        train_loader, test_loader, classes: Data loaders and a tuple of valid classes.
    """
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if dataset == 'cifar10':
        # Load CIFAR10 dataset
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)
    elif dataset == 'filters':
        train_set = FilterDataset('alexnet', './filters', is_training=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)

        test_set = FilterDataset('alexnet', './filters', is_training=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)

        classes = []
    elif dataset == 'random':
        train_set = RandomDataset('./data/random.hdf5', is_training=True)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=True, num_workers=num_workers)

        test_set = RandomDataset('./data/random.hdf5', is_training=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers)
    else:
        raise NotImplementedError('Invalid dataset: {}'.format(dataset))

    return train_loader, test_loader, classes


if __name__ == '__main__':
    model_dict = {
        'alexnet': models.alexnet,
        'resnet50': models.resnet50,
        'vgg19': models.vgg19_bn,
        'fd': models.filter_discriminator
    }

    filter_dict = {
        'alexnet': 'module.features.0',
        'resnet50': 'module.conv1',
        'vgg19': 'module.features.0'
    }

    parser = ArgParser()
    args_ = parser.parse_args()

    model_fn = model_dict[args_.model]
    model_ = model_fn(num_classes=1 if args_.model == 'fd' else 10)
    train_classifier(args_, model_)
