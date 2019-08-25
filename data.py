import os
import cv2
import tarfile
import numpy as np
import cPickle as pkl


class Images(object):
    # A class for easy and fast reading of images packed in a tar file
    def __init__(self, path, index_path=None):
        if index_path is None:
            # index file is the same as tar path but  .pkl
            index_path = path[:-3] + 'pkl'
        if not os.path.exists(index_path):
            print 'Indexing tar file, this could take a few minutes...'
            self._tar_index = self._index_tar(path)
            print 'done'
            # Save index file
            with open(index_path, 'wb') as fid:
                pkl.dump(self._tar_index, fid)
        else:
            with open(index_path, 'rb') as fid:
                self._tar_index = pkl.load(fid)
        self.fid = open(path, 'rb')

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
        return self._tar_index.keys()

    def __getitem__(self, path):
        # Grab an image buffer based on its path and decode it
        offset, size = self._tar_index[path]
        self.fid.seek(offset)
        buff = self.fid.read(size)
        image = self._decode_image(buff)[:, :, ::-1]
        return image

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.fid.close()


def read_pose(pose_path):
    # Read the pose points from file
    data = None
    with open(pose_path, 'rb') as fid:
        data = pkl.load(fid)
    keypoints = data['keypoints']
    scores = data['scores']
    paths = data['paths']
    return paths, keypoints, scores


def read_signatures(sigs_path):
    # Read the imagenet signatures from file
    data = None
    with open(sigs_path, 'rb') as fid:
        data = pkl.load(fid)
    signatures = data['signatures']
    paths = data['paths']
    return paths, signatures
