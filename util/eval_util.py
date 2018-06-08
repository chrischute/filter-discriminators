import math


def get_tp_score(val_loss, best_val_loss, num_classes=10):
    """Get train-percentage (TP) score for a given validation loss."""
    random_loss = math.log(num_classes)

    return (random_loss - val_loss) / (random_loss - best_val_loss)
