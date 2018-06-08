import itertools
import os
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from time import time
from datetime import datetime
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter


class TestLogger(object):
    """Class for logging test info to the console and saving test outputs to disk."""
    def __init__(self, args):
        super(TestLogger, self).__init__()
        self.epoch_start_time = None

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.save_dir, 'log.txt')

        log_dir = os.path.join(args.log_dir, '_'.join([args.name, datetime.now().strftime('%b%d_%H%M'), 'val']))
        self.summary_writer = SummaryWriter(log_dir=log_dir)
        self.y_true_buckets = []
        self.y_pred_buckets = []
        self.num_buckets = 4

    def start_iter(self):
        """Log info for start of an iteration."""
        pass

    def end_iter(self, filters, targets, logits):
        """Log info for end of an iteration."""
        probs = F.sigmoid(logits.detach())
        tgts = targets.detach()

        batch_size = filters.size(0)
        for i in range(batch_size):
            t = tgts[i].item()
            p = probs[i].item()
            f = filters[i]

            # Normalize filter values into 0, 1
            f -= f.min()
            f /= f.max()

            # Move channels last
            f.transpose(0, 1)
            f.transpose(1, 2)

            self.summary_writer.add_image('actual_{:.1f}/pred_{:.1f}'.format(t, p), f)

            self.y_pred_buckets.append(p * self.num_buckets // 1)
            self.y_true_buckets.append(t * self.num_buckets // 1)

    def _confusion_png(self, cm, classes, normalize=True, cmap=plt.cm.Blues):
        """Get NumPy array of confusion matrix plot."""
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-5)

        fig = plt.figure()

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title('TP Score Confusion (Val.)')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout(pad=2)
        plt.ylabel('True Quartile')
        plt.xlabel('Predicted Quartile')

        # If we haven't already shown or saved the plot, then we need to
        # draw the figure first...
        fig.canvas.draw()

        # Now we can save it to a numpy array.
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        return data

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.write('[start of test: writing to {}]'.format(self.save_dir))

    def end_epoch(self):
        """Log info for end of an epoch."""
        self.write('[end of test, time: {:.2g}]'.format(time() - self.epoch_start_time))

        confusion = confusion_matrix(self.y_true_buckets, self.y_pred_buckets, labels=list(range(self.num_buckets)))
        np.set_printoptions(precision=2)
        classes = [str(i) for i in range(self.num_buckets)]
        confusion_png = self._confusion_png(confusion, classes, normalize=False)
        self.summary_writer.add_image('confusion', confusion_png)
        self.summary_writer.file_writer.flush()

    def write(self, message, print_to_stdout=True):
        """Write a message to the log. If print_to_stdout is True, also print to stdout."""
        with open(self.log_path, 'a') as log_file:
            log_file.write(message + '\n')
        if print_to_stdout:
            print(message)
