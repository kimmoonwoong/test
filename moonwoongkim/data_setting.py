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
urbanlabelsetting ={}
filename_list = []
label_list = []
data_source = []
changelabel_list = []
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
    label_setting_fsd = {"Bark": 2, "Meow" : 3}
    FSD_train_dataset_path = 'C:\\Users\\user\\Desktop\\FSDKaggle2018.audio_train'
    a = 0
    for index, row in dataset.iterrows():
        file_name = FSD_train_dataset_path + '\\' + str(row["fname"])
        class_label = row["label"]
        if class_label in label_setting_fsd:
            data,sr = librosa.load(file_name, sr=16000)
            train_set.append([data, label_setting_fsd[class_label]])
            a += 1

        if a > 100: break
    print("데이터 생성 완료")
    return train_set
def UrBanDataset(train_set):
    Urbandataset = pd.read_csv('D:\\UrbanSound8K\\UrbanSound8K\\metadata\\UrbanSound8K.csv')
    Urban_train_dataset_path = 'D:\\UrbanSound8K\\UrbanSound8K\\audio'
    urban_data_label = {}
    label_setting_UrBan = {"car_horn": 1, "dog_bark" : 2, "siren" : 3}
    for index, row in Urbandataset.iterrows():
        class_label = row['class']
        if class_label in label_setting_UrBan:
            file_name = row['slice_file_name']
            fold_number = row['fold']
            file_path = Urban_train_dataset_path + '\\' + 'fold' + str(fold_number) + '\\' + file_name
            data, sr = librosa.load(file_path, sr = 16000)
            train_set.append([data, label_setting_UrBan[class_label]])
            # filename_list.append(file_name)
            # label_list.append(class_label)
            # data_source.append('UrBan')
            # changelabel_list.append(label_setting_UrBan[class_label])
        # if class_label in urban_data_label:
        #     urban_data_label[class_label] = urban_data_label.get(class_label) + 1
        # else:
        #     urban_data_label[class_label] = 1
    print("데이터 생성 완료")

    return train_set

def AI_HubDataset(train_set):
    Ai_Hub_dataset_path = 'D:\\도시소리'
    Ai_Hub_type_path = ['자동차', '이륜자동차', '동물']
    Ai_Hub_labelset = 'C:\\Users\\user\\Desktop\\교통소음'
    label_setting_aihub = {"차량경적": 1, "차량주행음" : 8, "차량사이렌" : 3, "이륜차경적": 1, "이륜차주행음" : 8, "개" : 2, "고양이" : 3}
    data_lock_count = {}
    for s in Ai_Hub_type_path:
        path = Ai_Hub_labelset + '\\' + str(s)
        filelist = os.listdir(path)
        for filename in filelist:
            file_path = path + '\\' + str(filename)
            class_label = ""
            with open(file_path, 'rt', encoding='UTF8') as file:
                jsondata = json.load(file)
                data_file_path = Ai_Hub_dataset_path + '\\' + str(jsondata["annotations"][0]["categories"]["category_01"]) + "\\" + str(jsondata["annotations"][0]["categories"]["category_02"]) + "\\" + str(jsondata["annotations"][0]["categories"]["category_03"]) + "\\" + str(jsondata["annotations"][0]["labelName"])
                class_label = str(jsondata["annotations"][0]["categories"]["category_03"])
                if class_label not in data_lock_count:
                    data_lock_count[class_label] = 1;
                else:
                    data_lock_count[class_label] = data_lock_count[class_label] + 1
                if(data_lock_count[class_label] >= 100): continue
                data, sr = librosa.load(data_file_path, sr = 16000)
                train_set.append([data, label_setting_aihub[class_label]])
                # filename_list.append(str(jsondata["annotations"][0]["labelName"]))
                # label_list.append(class_label)
                # data_source.append('Ai_Hub_도시소리데이터')
                # changelabel_list.append(label_setting_aihub[class_label])
            #data = librosa.load(Ai_Hub_dataset_path, sr = 16000)
            #train_set.append([data, class_label])

    print("데이터 생성 완료")
    return train_set


def slice_data(data, mins):
    slice_data = []
    for i in data:
        slice_data.append(i[:mins])

    slice_data = np.array(slice_data)

    return slice_data

def maxsize_data(data, maxs):

    for i in data:
        if(len(i) < maxs):
            print(maxs - len(i))
            for j in range(maxs - len(i)):
                item = np.array(0)
                i = np.append(i, item)
                print(j)
        print(i.shape)


def minlen(data):
    mins = 1000000
    for i in data:
        if len(i) < mins:
            mins = len(i)
    return mins

def maxlen(data):
    maxs = 0
    for i in data:
        if len(i) > maxs:
            maxs = len(i)

    return maxs
def extract_feature(data):
    mfccs = []
    slice = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))
    a = 0;
    for i in data:
        mfcc = librosa.feature.mfcc(y=i, sr=16000, n_mfcc=40, n_fft = 400)
        mfcc = slice(mfcc, 500)
        mfccs.append(mfcc)
        a+=1
        print(a)
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



train_data_set = []
train_data_set = FSDDataset(train_data_set)
train_data_set = UrBanDataset(train_data_set)
train_data_set = AI_HubDataset(train_data_set)

train_data_set = pd.DataFrame(train_data_set, columns=['data','label'])
train_x = np.array(train_data_set.data)

trains_mfcc = extract_feature(train_x)
trains_mfcc = np.array(trains_mfcc)
trains_mfcc = trains_mfcc.reshape(-1, trains_mfcc.shape[1], trains_mfcc.shape[2], 1)
print(np.array(trains_mfcc).shape)
train_X = trains_mfcc[:1900]
vail_X = trains_mfcc[1900:]

train_y = train_data_set.label[:1900]
vail_y = train_data_set.label[1900:].reset_index(drop=True)