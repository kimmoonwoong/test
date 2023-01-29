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
    a = 0
    for index, row in dataset.iterrows():
        file_name = train_dataset_path + '\\' + str(row["fname"])
        class_label = row["label"]
        data = librosa.load(file_name, sr=16000)
        train_set.append([data, class_label])
        a+=1
        if a > 100: break
    print("데이터 생성 완료")
    return pd.DataFrame(train_set, columns=['data','label'])

def test_set():
    testdatasetpath = 'C:\\Users\\user\\Desktop\\FSDKaggle2018.audio_test'
    test_set = []
    a = 0
    for index,row in testdataset.iterrows():
        file_name = testdatasetpath + '\\' + str(row['fname'])
        class_label = row['label']
        data = librosa.load(file_name, sr = 16000)
        test_set.append([data, class_label])
        a+=1
        if a > 100: break
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
        if len(i[0]) < mins:
            mins = len(i)
    return mins

def extract_feature(data):
    mfccs = []
    for i in data:
        mfcc = librosa.feature.mfcc(y=i, sr=16000, n_mfcc=40)
        mfcc = np.array(mfcc)
        print(mfcc.shape)
        mfccs.append(mfcc)
    return mfccs

def create_Data(isdataType):
    train_wav = train_set()
    test_wav = test_set()


    train_x = np.array(train_wav.data)
    test_x = np.array(test_wav.data)

    train_min = minlen(train_x)
    test_min = minlen(test_x)

    mins = np.min([train_min, test_min])
    train_x = slice_data(train_x, mins)
    test_x = slice_data(test_x, mins)

    trains_mfcc = extract_feature(train_x)
    trains_mfcc = np.array(trains_mfcc)

    trains_mfcc = trains_mfcc.reshape(-1, trains_mfcc.shape[1], trains_mfcc.shape[2], 1)
    return trains_mfcc

