#!/usr/bin/env python
import os
import numpy as np
import cPickle as pkl
from IPython import embed


def cosine_similarity(a, b):
    # Compute the cosine similarity between all vectors in a and b [NxC]
    _a = a / np.sqrt(np.sum(np.square(a), axis=1, keepdims=True))
    _b = b / np.sqrt(np.sum(np.square(b), axis=1, keepdims=True))
    return _a.dot(_b.T)


def enumerate_paths(paths):
    # Extract sequences/channels/people from the frame-paths
    sequences = [os.path.dirname(p) for p in paths]
    channels = [os.path.dirname(s) for s in sequences]
    people = [os.path.dirname(c) for c in channels]

    # Enumerate the frames based on channels and people
    unique_channels, channel_ids = np.unique(channels, return_inverse=True)
    unique_people, person_ids = np.unique(people, return_inverse=True)
    return person_ids, channel_ids


def train_test_split(person_ids, channel_ids, train_to_test_ratio=0.5):
    # Splits the channels of each person to train/test according to the train_to_test_ratio

    # Find borders where person id changes
    sections = np.where(np.diff(person_ids, 1))[0] + 1
    # Channels split by person id
    person_channels = np.split(channel_ids, sections)
    # Indices split by person id
    frame_indices = np.split(np.arange(len(person_ids)), sections)

    # Split channels train and test according to the train_to_test_ratio
    train_indices = []
    test_indices = []
    for pid, cids, fidx in zip(person_ids, person_channels, frame_indices):
        split_index = train_to_test_ratio * (cids[-1] - cids[0]) + cids[0]
        is_train = cids <= split_index
        train_indices.append(fidx[is_train])
        test_indices.append(fidx[~ is_train])
    train_indices = np.hstack(train_indices)
    test_indices = np.hstack(test_indices)
    assert len(set(train_indices).intersection(set(test_indices))) == 0
    return train_indices, test_indices


def split_by(data, indices):
    sections = np.where(np.diff(indices))[0] + 1
    split_data = np.split(data, sections)
    return split_data


def mean_signatures(signatures, indices):
    # Compute the mean signaures for each set of indices
    mean_signatures = np.vstack([np.mean(signatures[idx], axis=0)
                                 for idx in indices])
    return mean_signatures


def compute_accuracy(similarity_matrix, test_labels, verbose=True):
    # Compute and display top 1 / 5 accuracies
    num_classes = similarity_matrix.shape[1]
    gt_scores = similarity_matrix[range(len(similarity_matrix)), test_labels]
    rank = np.sum(similarity_matrix >= gt_scores[:, np.newaxis], axis=1)
    top1_accuracy = np.mean(rank == 1) * 100
    top5_accuracy = np.mean(rank <= 5) * 100
    if verbose:
        print 'top 1 accuracy {:.2f}% (naive: {:.2f}%)'.format(
            top1_accuracy, 100. / num_classes)
        print 'top 5 accuracy {:.2f}% (naive: {:.2f}%)'.format(
            top5_accuracy, 500. / num_classes)
    return top1_accuracy, top5_accuracy


def main(sigs_path, train_to_test_ratio=0.5):
    # Read the imagenet signatures from file
    with open(sigs_path, 'rb') as fid:
        data = pkl.load(fid)
    signatures = data['signatures']
    paths = data['paths']

    # Enumerate the frame paths based on person and channel
    person_ids, channel_ids = enumerate_paths(paths)
    # For each person, split his set of channels to train and test
    train_indices, test_indices = train_test_split(person_ids, channel_ids,
                                                   train_to_test_ratio)

    # Solution

    # Find the mean signature for each person based on the training set
    train_sigs = split_by(signatures[train_indices], person_ids[train_indices])
    train_sigs = np.vstack([np.mean(ts, axis=0) for ts in train_sigs])

    # Find the mean signature for each test - channel and assign its ground-truth person id
    test_sigs = split_by(signatures[test_indices], channel_ids[test_indices])
    test_sigs = np.vstack([np.mean(ts, axis=0) for ts in test_sigs])
    # Ground truth labels
    test_labels = np.array([pids[0] for pids in
                            split_by(person_ids[test_indices], channel_ids[test_indices])])

    # Predict classes using cosine similarity
    similarity_matrix = cosine_similarity(test_sigs, train_sigs)

    # Compute and display top 1 / 5 accuracies
    compute_accuracy(similarity_matrix, test_labels)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        raise ValueErrur('Missing signatures path')
    main(sys.argv[1])
