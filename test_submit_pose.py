# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 00:45:27 2019

@author: noasi
"""
import os
import numpy as np
import pickle as pkl
from data import read_pose
from utils import enumerate_paths
from utils import split_by
from evaluate import evaluate
from evaluate import submit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---- eval RNN on test data -----
 
def main(pose_path, submission_path, train_to_val_ratio=0.5):
    # Read the imagenet signatures from file
    paths_test, test_pose , test_scores= read_pose(pose_path)
    test_pose=test_pose[:,:13,:]
    test_pose=np.reshape(test_pose,(test_pose.shape[0],26))
   
    seq_ids = np.array([int(p.split('/')[0][4:]) for p in paths_test])
    test_seq_pose = split_by(test_pose, seq_ids)
   
    print(test_seq_pose.__len__())
    sequence_sz = 20 #20
 
    test_pose = test_seq_pose
    test_pose2 = []
    for testpose in test_pose:
        test_pose2.append(testpose[np.random.choice(testpose.shape[0], sequence_sz)])
 
    X_test = np.array(test_pose2)
 
    # Load in model and evaluate on validation data
    model = load_model('modelPoseRNN.h5')
    preds = model.predict(X_test)
 
    # Crate a submission - a sorted list of predictions, best match on the left.
    ranking = preds.argsort(axis=1)
    submission = [line.tolist() for line in ranking[:, :-6:-1]]
    print(submission[:10])
 
    from evaluate import submit
    submit('rrr', submission)
 
 
if __name__ == '__main__':
    import argparse
   
    args = {'pose_path':'data/pose_test.pkl',
       'submission_path':'submission.csv',
       'train_to_val_ratio':0.7}
 
    main(**args)