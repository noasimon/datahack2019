from __future__ import print_function

import os
import cv2
import tarfile
import numpy as np
import pickle as pkl


class Images(object):
    # A class for easy and fast reading of images packed in a tar file
    def __init__(self, path, index_path=None):
        if index_path is None:
            # index file is the same as tar path but  .pkl
            index_path = path[:-3] + 'pkl'
        if not os.path.exists(index_path):
            print('Indexing tar file, this could take a few minutes...')
            self._tar_index = self._index_tar(path)
            print('done')
            # Save index file
            with open(index_path, 'wb') as fid:
                pkl.dump(self._tar_index, fid)
        else:
            with open(index_path, 'rb') as fid:
                self._tar_index = pkl.load(fid)
        self.fid = open(path, 'rb')
        self.keys = sorted(self._tar_index.keys())

    @staticmethod
    def _index_tar(path):
        # Build a dictionary with the locations of all data points
        tar_index = {}
        with tarfile.TarFile(path, "r") as tar:
            for tarinfo in tar:
                offset_and_size = (tarinfo.offset_data, tarinfo.size)
                tar_index[tarinfo.name] = offset_and_size
        return tar_index

    @staticmethod
    def _decode_image(buff):
        # Decode an image buffer from memory
        buff_array = np.asarray(bytearray(buff), dtype='uint8')
        image = cv2.imdecode(buff_array, cv2.IMREAD_UNCHANGED)
        return image

    def __len__(self):
        return len(self._tar_index)

    @property
    def paths(self):
        return self.keys

    def __getitem__(self, item):
        if isinstance(item, int):
            item = self.keys[item]
        # Grab an image buffer based on its path and decode it
        offset, size = self._tar_index[item]
        self.fid.seek(offset)
        buff = self.fid.read(size)
        image = self._decode_image(buff)[:, :, ::-1]
        return image

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.fid.close()


def compatible_load(path):
    # pickle loading compatible for pyton 2/3
    data = None
    with open(path, 'rb') as fid:
        try:
            data = pkl.load(fid)
        except UnicodeDecodeError:
            # Python 3 compatability
            fid.seek(0)
            data = pkl.load(fid, encoding='latin1')
    return data


def read_pose(pose_path):
    # Read the pose points from file
    data = compatible_load(pose_path)
    keypoints = data['keypoints']
    scores = data['scores']
    paths = data['paths']
    return paths, keypoints, scores


def read_signatures(sigs_path):
    # Read the imagenet signatures from file
    data = compatible_load(sigs_path)
    signatures = data['signatures']
    paths = data['paths']
    return paths, signatures
