import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import librosa
import librosa.display
import os
import json
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plts
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics

dataset = pd.read_csv('C:\\Users\\user\\OneDrive - koreatech.ac.kr\\FSDKaggle2018.meta\\train_post_competition.csv')
testdataset = pd.read_csv('C:\\Users\\user\\Desktop\\FSDKaggle2018.meta\\test_post_competition_scoring_clips.csv')
Urbondataset = pd.read_csv('C:\\Users\\user\\OneDrive - koreatech.ac.kr\\UrbanSound8K\\UrbanSound8K\\metadata\\UrbanSound8K.csv')
urbanlabelsetting = {}
filename_list = []
label_list = []

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def FSDDataset(train_set):          # wav데이터 변환
    label_setting_fsd = {"Bark": 2} # 새롭게 클래스 라벨링
    FSD_train_dataset_path = 'C:\\Users\\user\\OneDrive - koreatech.ac.kr\\FSDKaggle2018.audio_train'
    data_lock_count = {} # 테스트용으로 각 라벨마다 100개씩 데이터를 뽑아 냄
    for index, row in dataset.iterrows():
        file_name = FSD_train_dataset_path + '\\' + str(row["fname"])
        class_label = row["label"]
        if class_label in label_setting_fsd:  # 데이터 셋에 우리가 쓸 데이터를 골라내는 작업

            label_check = []            # 나중에 학습할 때 loss계산을 위해 라벨링 형태를 맞춰줌
            for j in range(0, 8):
                if j + 1 == label_setting_fsd[class_label]:
                    label_check.append(1)
                else:
                    label_check.append(0)
            data, sr = librosa.load(file_name, sr=22050)    # librosa 모델을 사용하여 fft
            train_set.append([data, label_check])   # 데이터 저장


    print("데이터 생성 완료")
    return train_set


def UrBanDataset(train_set):    # wav파일 변환
    Urbandataset = pd.read_csv('C:\\Users\\user\\OneDrive - koreatech.ac.kr\\UrbanSound8K\\UrbanSound8K\\metadata\\UrbanSound8K.csv')
    Urban_train_dataset_path = 'C:\\Users\\user\\OneDrive - koreatech.ac.kr\\UrbanSound8K\\UrbanSound8K\\audio'
    label_setting_UrBan = {"car_horn": 1, "dog_bark": 2, "siren": 3}    # 새롭게 라벨링
    for index, row in Urbandataset.iterrows():
        class_label = row['class']
        if class_label in label_setting_UrBan:  # 우리가 쓸 데이터를 골라내는 작업
            file_name = row['slice_file_name']
            fold_number = row['fold']
            file_path = Urban_train_dataset_path + '\\' + 'fold' + str(fold_number) + '\\' + file_name
            label_check = []        # 나중에 학습할 때 loss계산을 위해 라벨링 형태를 맞춰줌
            for j in range(0, 6):
                if j + 1 == label_setting_UrBan[class_label]:
                    label_check.append(1)
                else:
                    label_check.append(0)
            data, sr = librosa.load(file_path, sr=22050) # librosa 모델을 사용하여 fft
            train_set.append([data, label_check])   # 데이터 저장

    print("데이터 생성 완료")

    return train_set

def AI_HubDataset(train_set):
    Ai_Hub_dataset_path = 'C:\\Users\\user\\OneDrive - koreatech.ac.kr\\도시소리'
    Ai_Hub_type_path = ['자동차', '이륜자동차', '동물']
    Ai_Hub_labelset = 'C:\\Users\\user\\OneDrive - koreatech.ac.kr\\교통소음'
    label_setting_aihub = {"차량경적": 1, "차량사이렌": 3, "이륜차경적": 1, "개": 2} # 새롭게 라벨링
    for s in Ai_Hub_type_path:
        path = Ai_Hub_labelset + '\\' + str(s)
        filelist = os.listdir(path)
        for filename in filelist:
            file_path = path + '\\' + str(filename)
            class_label = ""
            with open(file_path, 'rt', encoding='UTF8') as file:
                jsondata = json.load(file)
                data_file_path = Ai_Hub_dataset_path + '\\' + str(
                    jsondata["annotations"][0]["categories"]["category_01"]) + "\\" + str(
                    jsondata["annotations"][0]["categories"]["category_02"]) + "\\" + str(
                    jsondata["annotations"][0]["categories"]["category_03"]) + "\\" + str(
                    jsondata["annotations"][0]["labelName"])
                class_label = str(jsondata["annotations"][0]["categories"]["category_03"])
                if class_label in label_setting_aihub: # 우리가 쓸 데이터를 뽑아 냄
                    label_check = []    # 나중에 학습할 때 loss계산을 위해 라벨링 형태를 맞춰줌
                    for j in range(0, 6):
                        if j + 1 == label_setting_aihub[class_label]:
                            label_check.append(1)
                        else:
                            label_check.append(0)
                    data, sr = librosa.load(data_file_path, sr=22050)   # librosa 모델을 사용하여 fft
                    train_set.append([data, label_check])   # 데이터 저장


    print("데이터 생성 완료")
    return train_set

def AI_HubAlertDataset(train_set):
    AI_HubAlert_dataset_path = 'C:\\Users\\user\\OneDrive - koreatech.ac.kr\\경보소리'
    AI_HubAlert_type_path = ["도난경보", "화재경보", "비상경보"]
    AI_HubAlert_label_path = "C:\\Users\\user\\OneDrive - koreatech.ac.kr\\경보소리라벨링\\경보"
    label_setting_ai_hubAleart = {"도난경보 소리": 6, "도난 경보음 소리": 6, "침입감지 경보 소리": 6, "화재경보 소리":5, "화재 경보 소리": 5,
                                  "가스누설 화재경보 소리": 5, "자동차 경적 소리": 1, "비상경보 소리": 4, "철도 건널목 신호음 소리": 4, "민방위훈련 사이렌 소리": 4, "공습경보 소리" : 4} # 새롭게 라벨링
    for s in AI_HubAlert_type_path:
        print(s)
        path = AI_HubAlert_label_path + "\\" + str(s)
        fill_list = os.listdir(path)
        for filename in fill_list:
            class_label = ""
            file_path = path + "\\" + filename
            with open(file_path, 'rt', encoding='UTF8') as file:
                jsondata = json.load(file)
                data_file_path = AI_HubAlert_dataset_path + "\\" + str(s) + "\\" + str(
                    jsondata["LabelDataInfo"]["LabelID"]) + ".wav"
                class_label = str(jsondata["LabelDataInfo"]["Desc"])

                label_check = []        # 나중에 학습할 때 loss계산을 위해 라벨링 형태를 맞춰줌
                for j in range(0, 6):
                    if j + 1 == label_setting_ai_hubAleart[class_label]:
                        label_check.append(1)
                    else:
                        label_check.append(0)
                data, sr = librosa.load(data_file_path, sr=22050)       # librosa 모델을 사용하여 fft
                for i in range(0,10):
                    train_set.append([data, label_check])                   # 데이터 저장
    print("데이터 생성 완료")
    return train_set

train_data_label = []


def extract_feature(data, label, isCheck):      # 위에서 변환한 데이터를 mfcc의 이용으로 소리 데이터를 벡터화 시킴( 벡터화 시킴으로써 CNN모델을 사용할 수 있게 됨 )
    mfccs = []
    slice = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))    # 데이터의 길이를 설정한 길이에 맞게 맞춰줌
    index = 0           # 한 군데에서 가져온 데이터가 앞에 10가량이 무슨 소리인지 소개하는 소리이기에 짜르기 위해 그 데이터를 찾아냄
    for i in data:
        mfcc = librosa.feature.mfcc(y=i, sr=22050, n_mfcc=40, n_fft=400)    # mfcc를 통해 벡터화를 시킴
        if index >= isCheck:            # 앞서 말한 데이터부터 10초가량을 짜름
            mfcc = mfcc[:, 1100:]
        else:
            index += 1
        mfcc = slice(mfcc, 96)     # 설정한 길이에 맞게 맞춰주는 작업

        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mfcc2 = librosa.feature.delta(mfcc, order=2)
        features = np.concatenate([mfcc, delta_mfcc, delta_mfcc2], axis=0)
        mfccs.append(features)          # 데이터 저장
    return mfccs


train_data_set = []     # wav를 fft시키는 작업
train_data_set = FSDDataset(train_data_set)
train_data_set = UrBanDataset(train_data_set)
train_data_set = AI_HubDataset(train_data_set)
isCheck = len(train_data_set)
train_data_set = AI_HubAlertDataset(train_data_set)
print(len(train_data_set))
train_data_set = pd.DataFrame(train_data_set, columns=['data', 'label'])
# f = open("data_setcheck.csv", "w")
#
# for i in range(len(train_data_set)):
#     write = csv.writer(f)
#     write.writerows(str(train_data_set.data[i].tolist()))
# f.close()
train_x = np.array(train_data_set.data)
trains_mfcc = extract_feature(train_x, train_data_set.label, isCheck)   #fft된 데이터를 mfcc를 적용시켜 벡터화 시킴
print(len(trains_mfcc))
trains_mfcc = np.array(trains_mfcc)
trains_mfcc = trains_mfcc.reshape(-1, trains_mfcc.shape[1], trains_mfcc.shape[2], 1)
print(trains_mfcc.shape)
train_X = trains_mfcc
train_y = np.array(train_data_set.label)
for i in range(len(train_y)):   #학습을 위해 라벨링 데이터도 numpy형태로 변환
    train_y[i] = np.array(train_y[i])

temp = [[x,y] for x,y in zip(train_X,train_y)]  # 데이터를 랜덤으로 섞기 위해 데이터와 라벨링 데이터를 묶음
import random
random.shuffle(temp)        # 랜덤으로 돌림

train_X = [n[0] for n in temp]     #다시 데이터와 라벨링으로 나눔
train_y = [n[1] for n in temp]
# 훈련 셋과 검증 셋으로 7:3으로 나눔

test_data_X = train_X[len(train_X) - 12:]
test_data_y = train_y[len(train_X) - 12:]

train_X = train_X[:len(train_X) - 12]
train_y = train_y[:len(train_y) - 12]
from keras import layers
from keras.layers import LSTM
model = keras.Sequential()
model.add(layers.LSTM(256, input_shape=(120, 1), return_sequences=False))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(6, activation="softmax"))

model.summary()
batch_size = 6
num_epoc = 100

vail_X = train_X[34745:]
vail_y = train_y[34745:]
train_X = train_X[:34745]
train_y = train_y[:34745]

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])
history = model.fit(train_X, train_y, batch_size=batch_size, epochs=num_epoc)

model.save("C:\\Users\\user\\Desktop\\ai\\mobail_model_2layer")
ans = model.evaluate(test_data_X, test_data_y, batch_size=6)
print(ans)