import csv
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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def FSDDataset(train_set):
    train_data_label = {}
    label_setting_fsd = {"Bark": 2, "Meow" : 3, "Bus": 4, "Squeak": 4, "Knock" : 4}
    FSD_train_dataset_path = 'C:\\Users\\user\\Desktop\\FSDKaggle2018.audio_train'
    a = 0
    data_lock_count = {}
    for index, row in dataset.iterrows():
        file_name = FSD_train_dataset_path + '\\' + str(row["fname"])
        class_label = row["label"]
        if class_label in label_setting_fsd:


            if class_label not in data_lock_count:
                data_lock_count[class_label] = 1
            elif data_lock_count[class_label] >= 100:
                continue
            else:
                data_lock_count[class_label] = data_lock_count[class_label] + 1

            label_check = []
            for j in range(0, 4):
                if j + 1 == label_setting_fsd[class_label]:
                    label_check.append(1)
                else:
                    label_check.append(0)
            data,sr = librosa.load(file_name, sr=16000)
            train_set.append([data, label_check])

        a = 0
        for k in data_lock_count.values():
            a+=k
        if a >= 500:
            break

    print("데이터 생성 완료")
    return train_set
def UrBanDataset(train_set):
    Urbandataset = pd.read_csv('D:\\UrbanSound8K\\UrbanSound8K\\metadata\\UrbanSound8K.csv')
    Urban_train_dataset_path = 'D:\\UrbanSound8K\\UrbanSound8K\\audio'
    urban_data_label = {}
    data_lock_count = {}
    label_setting_UrBan = {"car_horn": 1, "dog_bark" : 2, "siren" : 3, "street_music": 4, "drilling": 4, "air_conditioner" : 4, "jachammer": 4}
    for index, row in Urbandataset.iterrows():
        class_label = row['class']
        if class_label in label_setting_UrBan:
            file_name = row['slice_file_name']
            fold_number = row['fold']
            file_path = Urban_train_dataset_path + '\\' + 'fold' + str(fold_number) + '\\' + file_name


            if class_label not in data_lock_count:
                data_lock_count[class_label] = 1
            elif data_lock_count[class_label] >= 100:
                continue
            else:
                data_lock_count[class_label] = data_lock_count[class_label] + 1

            label_check = []
            for j in range(0, 4):
                if j + 1 == label_setting_UrBan[class_label]:
                    label_check.append(1)
                else:
                    label_check.append(0)
            data, sr = librosa.load(file_path, sr = 16000)
            train_set.append([data, label_check])

        a = 0
        for k in data_lock_count.values():
            a += k
        if a >= 700:
            break
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
    label_setting_aihub = {"차량경적": 1, "차량주행음" : 4, "차량사이렌" : 3, "이륜차경적": 1, "이륜차주행음" : 4, "개" : 2, "고양이" : 3}
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
                if class_label in label_setting_aihub:

                    if class_label not in data_lock_count:
                        data_lock_count[class_label] = 1
                    elif data_lock_count[class_label] >= 100:
                        continue
                    else:
                        data_lock_count[class_label] = data_lock_count[class_label] + 1
                    label_check = []
                    for j in range(0, 4):
                        if j + 1 == label_setting_aihub[class_label]:
                            label_check.append(1)
                        else:
                            label_check.append(0)
                    data, sr = librosa.load(data_file_path, sr = 16000)
                    train_set.append([data, label_check])

        a = 0
        for k in data_lock_count.values():
            a += k
        if a >= 700:
            break
                # filename_list.append(str(jsondata["annotations"][0]["labelName"]))
                # label_list.append(class_label)
                # data_source.append('Ai_Hub_도시소리데이터')
                # changelabel_list.append(label_setting_aihub[class_label])
            #data = librosa.load(Ai_Hub_dataset_path, sr = 16000)
            #train_set.append([data, class_label])

    print("데이터 생성 완료")
    return train_set
train_data_label = []
def extract_feature(data, label):
    mfccs = []
    slice = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))
    index = 0
    for i in data:
        mfcc = librosa.feature.mfcc(y=i, sr=16000, n_mfcc=40, n_fft=400)

        mfcc = slice(mfcc, 400)
        mfccs.append(mfcc)
    return mfccs



train_data_set = []
train_data_set = FSDDataset(train_data_set)
train_data_set = UrBanDataset(train_data_set)
train_data_set = AI_HubDataset(train_data_set)


train_data_set = pd.DataFrame(train_data_set, columns=['data','label'])
# f = open("data_setcheck.csv", "w")
#
# for i in range(len(train_data_set)):
#     write = csv.writer(f)
#     write.writerows(str(train_data_set.data[i].tolist()))
# f.close()
train_x = np.array(train_data_set.data)

trains_mfcc = extract_feature(train_x, train_data_set.label)
trains_mfcc = np.array(trains_mfcc)
trains_mfcc = trains_mfcc.reshape(-1, trains_mfcc.shape[1], trains_mfcc.shape[2], 1)
print(np.array(trains_mfcc).shape)
print(len(trains_mfcc))
train_X = trains_mfcc[:1200]
vail_X = trains_mfcc[1200:]
train_y = np.array(train_data_set.label[:1200])
for i in range(len(train_y)):
    train_y[i] = np.array(train_y[i])
vail_y = np.array(train_data_set.label[1200:].reset_index(drop=True))
for i in range(len(vail_y)):
    vail_y[i] = np.array(vail_y[i])
print(len(train_X), " " , len(train_y))
print(len(vail_X), " " , len(vail_y))
class Custom_Dataset(Dataset):
    def __init__(self, X, y, train_mode=True, transforms=None):
        self.X = X
        self.y = y
        self.train_mode = train_mode
        self.transforms = transforms

    def __getitem__(self, index):
        X = self.X[index]

        if self.transforms is not None:
            X = self.transforms(X)

        if self.train_mode:
            y = self.y[index]
            return X,y
        else:
            return X
    def __len__(self):
        return len(self.X)

num_epochs = 150
batch_size = 6
train_dataset = Custom_Dataset(X=train_X, y = train_y)
train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True)

vail_dataset = Custom_Dataset(X=vail_X, y = vail_y)
vail_loader = DataLoader(vail_dataset, batch_size=batch_size, shuffle=True)




import torch.nn
import torch.nn as nn

class CNNclassification(torch.nn.Module):
    def __init__(self):
        super(CNNclassification, self).__init__()
        self.layer1 = torch.nn.Sequential(
            nn.Conv2d(40, 100, kernel_size=2, stride=1, padding=1),  # cnn layer
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

        self.layer2 = torch.nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=2, stride=1, padding=1),  # cnn layer
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

        self.layer3 = torch.nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=2, stride=1, padding=1),  # cnn layer
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

        self.layer4 = torch.nn.Sequential(
            nn.Conv2d(100, 100, kernel_size=2, stride=1, padding=1),  # cnn layer
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

        self.fc_layer = nn.Sequential(
            nn.Linear(2500, 4)  # fully connected layer(ouput layer)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.flatten(x, start_dim=1)
        out = self.fc_layer(x)
        return out

import torch.optim as optim
torch.cuda.empty_cache()
model = CNNclassification().to(device)
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3)
scheduler = None
print(vail_loader)

from tqdm.auto import tqdm


def train(model, optimizer, train_loader, scheduler, device):
    model.to(device)
    n = len(train_loader)
    best_acc = 0

    for epoch in range(1, num_epochs):  # 에포크 설정
        model.train()  # 모델 학습
        running_loss = 0.0

        for wav, label in tqdm(iter(train_loader)):
            wav, label = wav.to(device).float(), label.to(device).long()  # 배치 데이터
            optimizer.zero_grad()  # 배치마다 optimizer 초기화
            # Data -> Model -> Output
            logit = model(wav)  # 예측값 산출
            loss = criterion(logit, torch.max(label,1)[1])  # 손실함수 계산
            # 역전파
            loss.backward()  # 손실함수 기준 역전파
            optimizer.step()  # 가중치 최적화
            running_loss += loss.item()

        print('[%d] Train loss: %.10f' % (epoch, running_loss / len(train_loader)))

        if scheduler is not None:
            scheduler.step()

        # Validation set 평가
        model.eval()  # evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키도록 하는 함수
        vali_loss = 0.0
        correct = 0

        with torch.no_grad():  # 파라미터 업데이트 안하기 때문에 no_grad 사용
            for wav, label in tqdm(iter(vail_loader)):
                wav, label = wav.to(device).float(), label.to(device).long()
                logit = model(wav)
                vali_loss += criterion(logit, torch.max(label,1)[1])
                pred = logit.argmax(dim=1, keepdim=True)  # 10개의 class중 가장 값이 높은 것을 예측 label로 추출
                label = label.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()  # 예측값과 실제값이 맞으면 1 아니면 0으로 합산
        vali_acc = 100 * correct / len(vail_loader.dataset)
        print('Vail set: Loss: {:.4f}, Accuracy: {}/{} ( {:.0f}%)\n'.format(vali_loss / len(vail_loader), correct,
                                                                            len(vail_loader.dataset),
                                                                            100 * correct / len(vail_loader.dataset)))

        # 베스트 모델 저장
        if best_acc < vali_acc:
            best_acc = vali_acc
            torch.save(model.state_dict(),
                       'C:\\Users\\user\\Desktop\\ai\\best_model2.pth')  # 이 디렉토리에 best_model.pth을 저장
            print('Model Saved.')

train(model, optimizer, train_loader,scheduler,device)