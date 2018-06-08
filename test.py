import models
import torch.utils.data
import torch.nn as nn

from args import ArgParser
from datasets import FilterDataset
from logger import TestLogger
from saver import ModelSaver


def test(args):

    model_fn = model_dict[args_.model]
    model = model_fn(num_classes=1 if args_.model == 'fd' else 10)
    model = nn.DataParallel(model, args.gpu_ids)

    ckpt_info = ModelSaver.load_model(args.ckpt_path, model)
    args.start_epoch = ckpt_info['epoch'] + 1
    model = model.to(args.device)
    model.eval()

    test_set = FilterDataset('alexnet', './filters', is_training=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.num_workers)
    logger = TestLogger(args)

    logger.start_epoch()
    for inputs, labels in test_loader:
        logger.start_iter()

        with torch.set_grad_enabled(True):
            # Forward
            logits = model.forward(inputs.to(args.device))

        logger.end_iter(inputs, labels, logits)
    logger.end_epoch()


if __name__ == '__main__':
    model_dict = {
        'alexnet': models.alexnet,
        'resnet50': models.resnet50,
        'vgg19': models.vgg19_bn,
        'fd': models.filter_discriminator
    }

    parser = ArgParser()
    args_ = parser.parse_args()
    test(args_)
