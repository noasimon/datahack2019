# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:46:06 2019

@author: noasi
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:18:56 2019

@author: noasi
"""

#!/usr/bin/env python
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
  


def cosine_similarity(a, b):
    # Compute the cosine similarity between all vectors in a and b [NxC]
    _a = a / np.sqrt(np.sum(np.square(a), axis=1, keepdims=True))
    _b = b / np.sqrt(np.sum(np.square(b), axis=1, keepdims=True))
    return _a.dot(_b.T)


"""def mean_signatures(signatures, indices):
    # Compute the mean signaures for each set of indices
    mean_signatures = np.vstack([np.mean(signatures[idx], axis=0)
                                 for idx in indices])
    return mean_signatures
"""
def train_val_split(person_ids, video_ids, train_to_test_ratio=0.7):
    # Splits the videos of each person to train/test according to the train_to_test_ratio
 
    # Find borders where person id changes
    sections = np.where(np.diff(person_ids, 1))[0] + 1
    # videos split by person id
    person_videos = np.split(video_ids, sections)
    # Indices split by person id
    frame_indices = np.split(np.arange(len(person_ids)), sections)
 
    # Split videos train and test according to the train_to_test_ratio
    train_indices = []
    test_indices = []
    for pid, cids, fidx in zip(person_ids, person_videos, frame_indices):
        split_index = train_to_test_ratio * (cids[-1] - cids[0]) + cids[0]
        is_train = cids <= split_index
        train_indices.append(fidx[is_train])
        test_indices.append(fidx[~ is_train])
    train_indices = np.hstack(train_indices)
    test_indices = np.hstack(test_indices)
    assert len(set(train_indices).intersection(set(test_indices))) == 0
    return train_indices, test_indices
 
def main(pose_train, pose_test):
    # Read the poses from file
    #here train=train+val
    paths_train, train_pose, train_scores = read_pose(pose_train)
    paths_test, test_pose, test_scores = read_pose(pose_test)

    # Solution
    #using only the poses that have score > 0 in more than 50% of the frames (= the first 13 poses)
    #filtering good values of poses
    bad=np.sum(train_scores[:,:13]<0,1)
    good=bad<7
    good_indices=np.where(good)[0]
    good_paths_train=np.array(paths_train)[good_indices]
    #here t=train+val
    t1=train_pose[:,:13,:]
    t_pose=np.reshape(t1,(581685,26))
    t_pose=t_pose[good_indices,:]
   #train
    sequence_sz=20
    person_ids, video_ids = enumerate_paths(paths_train)
#    train_indices, val_indices = train_val_split(person_ids, video_ids, 0.7)
    train_indices, val_indices = train_val_split(person_ids[good_indices], video_ids[good_indices], 0.7)
    video_ids=video_ids[good_indices]
    
    train_p = split_by(t_pose[train_indices],video_ids[train_indices] )
    train_p2 = []
    for train_p1 in train_p:
        train_p2.append(train_p1[np.random.choice(train_p1.shape[0], sequence_sz)])
    X_train=np.array(train_p2)
    #val
    val_p = split_by(t_pose[val_indices], video_ids[val_indices])
    val_p2 = []
    for val_p1 in val_p:
        val_p2.append(val_p1[np.random.choice(val_p1.shape[0], sequence_sz)])
    X_val=np.array(val_p2)
    
    # Ground truth labels
    val_labels = np.array([pids[0] for pids in
                            split_by(person_ids[val_indices], video_ids[val_indices])])
    train_labels = np.array([pids[0] for pids in
                            split_by(person_ids[train_indices], video_ids[train_indices])])
    
    
    
    
    y_train = np.zeros((train_labels.shape[0], 101))
    y_train[np.arange(train_labels.shape[0]), train_labels] = 1
    y_valid = np.zeros((val_labels.shape[0], 101))
    y_valid[np.arange(val_labels.shape[0]), val_labels] = 1
    
    # --- build RNN model ---
    model = Sequential()
    # inp sz
    poseNum = 26
#     maxSequenceSz = sequence_sz  # =20, 30, ...3999
    batchSz = 32
    # Recurrent layer
    hiddenSz = 496
    model.add(LSTM(hiddenSz, return_sequences=False,
                   dropout=0.1, recurrent_dropout=0.1, input_shape=(sequence_sz, poseNum)))
    # Fully connected layer
    fullySz = 64
    model.add(Dense(fullySz, activation='relu'))
    # Dropout for regularization
    model.add(Dropout(0.5))
    # Output layer - 101 persons
    outSz = 101
    model.add(Dense(outSz, activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
    print(1)
    print(model.summary())
    # Create callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=5),
                 ModelCheckpoint('modelPoseRNN.h5', save_best_only=True, save_weights_only=False)]
    # fit
    history = model.fit(X_train, y_train, batch_size=batchSz, epochs=50, callbacks=callbacks,
                        validation_data=(X_val, y_valid))
 

# --- eval on validation data ---
    # Load in model and evaluate on validation data
    model = load_model('modelPoseRNN.h5')
    model.evaluate(X_val, y_valid)
    # test the performance on the validation data
    preds = model.predict(X_val)  
    # Crate a submission - a sorted list of predictions, best match on the left.
    ranking = preds.argsort(axis=1)
    submission = [line.tolist() for line in ranking[:, :-6:-1]]
    print(submission[:10]) 
   


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='solution_submit_pose')
    parser.add_argument(
        '--pose_train',  help='path for train pose pkl', default='data/pose.pkl')
    parser.add_argument(
        '--pose_test',  help='path for test pose pkl', default='data/pose_test.pkl')
    args = parser.parse_args()

    main(**vars(args))