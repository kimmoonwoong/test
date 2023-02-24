import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import librosa
import os
import glob
import pickle
import sklearn
dataset = pd.read_csv('C:\\Users\\user\\Desktop\\FSDKaggle2018.meta\\train_post_competition.csv')

max_pad_len = 174
def extract_feature(file_name):
    print('file name :', file_name)
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(audio,sr=sample_rate , n_mfcc=100, n_fft=550, hop_length=220)
        mfccs = sklearn.preprocessing.scale(mfccs,axis=1)
        pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i-a.shape[1]))))
        mfccs = pad2d(mfccs, 40)
        print(mfccs.shape)

    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        print(e)
        return None

    return mfccs


fulldatasetpath = 'C:\\Users\\user\\Desktop\\FSDKaggle2018.audio_train'
features = []
lables = {}
a = 0
# Iterate through each sound file and extract the features
for index, row in dataset.iterrows():
    file_name = fulldatasetpath + '\\' + str(row["fname"])
    class_label = row["label"]
    data = extract_feature(file_name)
    lables[class_label] = 1
    features.append([data, class_label])

print(len(features))
print(lables.keys())