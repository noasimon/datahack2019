import numpy as np


def evaluate(submission_path, test_labels, verbose=True):
    # Parse submission path
    with open(submission_path, 'rt') as fid:
        lines = fid.read().split('\n')
    result = np.array([map(int, line.split(',')) for line in lines])
    # Compute and display top 1 / 5 accuracies
    top1_accuracy = np.mean(result[:, 0] == test_labels) * 100
    top5_accuracy = np.mean(
        np.any(result == test_labels[:, np.newaxis], axis=1)) * 100
    if verbose:
        print 'top 1 accuracy {:.2f}%'.format(top1_accuracy)
        print 'top 5 accuracy {:.2f}%'.format(top5_accuracy)
    return top1_accuracy, top5_accuracy
