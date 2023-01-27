import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import librosa
import os
import glob
import pickle
import sklearn
from torch.utils.data import Dataset, DataLoader
dataset = pd.read_csv('C:\\Users\\user\\Desktop\\FSDKaggle2018.meta\\train_post_competition.csv')
testdataset = pd.read_csv('C:\\Users\\user\\Desktop\\FSDKaggle2018.meta\\test_post_competition_scoring_clips.csv')
def train_set():
    train_dataset_path = 'C:\\Users\\user\\Desktop\\FSDKaggle2018.audio_train'
    train_set = []

    for index, row in dataset.iterrows():
        file_name = train_dataset_path + '\\' + str(row["fname"])
        class_label = row["label"]
        data = librosa.load(file_name, sr=16000)
        train_set.append([data, class_label])

    print("데이터 생성 완료")
    return pd.DataFrame(train_set, columns=['data','label'])

def test_set():
    testdatasetpath = 'C:\\Users\\user\\Desktop\\FSDKaggle2018.audio_test'
    test_set = []

    for index,row in testdataset.iterrows():
        file_name = testdatasetpath + '\\' + str(row['fname'])
        class_label = row['label']
        data = librosa.load(file_name, sr = 16000)
        test_set.append([data, class_label])

    print("테스트 데이터 생성 완료")
    return pd.DataFrame(test_set, columns=['data', 'label'])

def slice_data(data, mins):
    slice_data = []
    for i in data:
        slice_data.append(i[:mins])

    slice_data = np.array(slice_data)

    return slice_data
def minlen(data):
    mins = 1000000
    for i in data:
        if len(i) < mins:
            mins = len(i)
    return mins

def extract_feature(data):
    mfccs = []
    for i in data:
        mfcc = librosa.feature.mfcc(y=i, sr = 16000, n_mfcc=40)
        mfccs.append(mfcc)
    return mfccs


train_wav = train_set()
test_wav = test_set()

train_x = np.array(train_wav.data)
test_x = np.array(test_wav.data)

train_data_minlen = minlen(train_x)
test_data_minlen = minlen(test_x)
mins = np.min([train_data_minlen,test_data_minlen])


train_x = slice_data(train_x, mins)
test_x = slice_data(test_x, mins)

print('train : ', train_x.shape)
print('test : ', test_x.shape)
trains_mfcc = extract_feature(train_x)
trains_mfcc = np.array(trains_mfcc)
trains_mfcc = trains_mfcc.reshape(-1, trains_mfcc[1], trains_mfcc[2], 1)

