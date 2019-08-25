import os
import numpy as np


def enumerate_paths(paths):
    # Extract sequences/videos/people from the frame-paths
    sequences = [os.path.dirname(p) for p in paths]
    videos = [os.path.dirname(s) for s in sequences]
    people = [os.path.dirname(c) for c in videos]

    # Enumerate the frames based on videos and people
    unique_videos, video_ids = np.unique(videos, return_inverse=True)
    unique_people, person_ids = np.unique(people, return_inverse=True)
    return person_ids, video_ids


def split_by(data, indices):
    # Split data based on a numpy array of sorted indices
    sections = np.where(np.diff(indices))[0] + 1
    split_data = np.split(data, sections)
    return split_data
