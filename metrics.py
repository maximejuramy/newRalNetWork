import numpy as np


def accuracy(true_labels, predictions):
    """
    Returns an accuracy (%) from true labels
    and predictions
    """
    return np.array(
        [true_labels == predictions]
    ).sum()/len(true_labels)*100
