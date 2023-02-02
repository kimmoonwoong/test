import json

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
import json
dataset = pd.read_csv('C:\\Users\\user\\Desktop\\FSDKaggle2018.meta\\train_post_competition.csv')
testdataset = pd.read_csv('C:\\Users\\user\\Desktop\\FSDKaggle2018.meta\\test_post_competition_scoring_clips.csv')
Urbondataset = pd.read_csv('D:\\UrbanSound8K\\UrbanSound8K\\metadata\\UrbanSound8K.csv')
label_setting_fsd = {["Bark", "Meow"]: 1, ["Shatter", ] : 2}
urbanlabelsetting ={}
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

def labelCheck():
    train_data_label = {}

    for index,row in dataset.iterrows():
        class_label = row['label']
        if class_label in train_data_label:
            train_data_label[class_label] = train_data_label.get(class_label) + 1
        else:
            train_data_label[class_label] = 1


    return train_data_label

def FSDDataset(train_set):
    train_data_label = {}
    label_setting_fsd = {["Bark", "Meow"]: 1, ["Shatter", ]: 2}
    FSD_train_dataset_path = 'C:\\Users\\user\\Desktop\\FSDKaggle2018.audio_train'
    a = 0
    for index, row in dataset.iterrows():
        file_name = FSD_train_dataset_path + '\\' + str(row["fname"])
        class_label = row["label"]
        data = librosa.load(file_name, sr=16000)
        train_set.append([data, class_label])
        a += 1
        if a > 100: break
    print("데이터 생성 완료")
    return train_set
def UrBanDataset():
    Urbandataset = pd.read_csv('D:\\UrbanSound8K\\UrbanSound8K\\metadata\\UrbanSound8K.csv')
    urban_data_label = {}

    for index, row in Urbandataset.iterrows():
        class_label = row['class']
        if class_label in urban_data_label:
            urban_data_label[class_label] = urban_data_label.get(class_label) + 1
        else:
            urban_data_label[class_label] = 1

    return urban_data_label

def AI_HubDataset():
    Ai_Hub_dataset_path = 'D:\\'
    Ai_Hub_type_path = {'교통소음' : ['자동차', '이륜자동차', '항공기', '열차'], '생활소음': ['충격', '가전', '동물', '도구'], '사업장소음': ['공사장', '공장']}
    Ai_Hub_labelset = 'C:\\Users\\user\\Desktop\\교통소음'

    for s in Ai_Hub_type_path:
        path = Ai_Hub_labelset + '\\' + str(s)
        filelist = os.listdir(path)
        train_set = []
        for filename in filelist:
            file_path = path + '\\' + str(filename)
            class_label = ""
            with open(file_path, 'rt', encoding='UTF8') as file:
                jsondata = json.load(file)
                print(data["annotations"][0]["categories"])
                Ai_Hub_dataset_path = Ai_Hub_dataset_path + str(jsondata["annotations"][0]["categories"]["category_01"]) + "\\" + str(jsondata["annotations"][0]["categories"]["category_02"]) + "\\" + str(jsondata["annotations"][0]["categories"]["category_03"]) + "\\" + str(jsondata["labelName"])
                class_label = str(jsondata["annotations"][0]["categories"]["category_03"])
            data = librosa.load(Ai_Hub_dataset_path, sr = 16000)
            train_set.append([data, class_label])


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



label = labelCheck()
print("FSD: ")
len(label)
for k in label.keys():
    print(k, " ", label[k])
print('------------------------------------------')
urbanlabel = UrBanDataset()
print("Urban: ")
len(urbanlabel)
for k in urbanlabel.keys():
    print(k, " ", urbanlabel[k])

path = "C:\\Users\\user\\Desktop\\1.자동차\\1.차량경적\\1.자동차_1.json"

with open(path, 'rt', encoding='UTF8') as file:
    data = json.load(file)
    print(data["annotations"][0]["categories"])
    print()


