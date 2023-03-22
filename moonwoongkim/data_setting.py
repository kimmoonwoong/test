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
from sklearn.model_selection import StratifiedKFold
import wandb

dataset = pd.read_csv('D:\\FSDKaggle2018.meta\\train_post_competition.csv')
testdataset = pd.read_csv('C:\\Users\\user\\Desktop\\FSDKaggle2018.meta\\test_post_competition_scoring_clips.csv')
Urbondataset = pd.read_csv('D:\\UrbanSound8K\\UrbanSound8K\\metadata\\UrbanSound8K.csv')
urbanlabelsetting = {}
filename_list = []
label_list = []

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def FSDDataset(train_set):          # wav데이터 변환
    label_setting_fsd = {"Bark": 2, "Meow": 3, "Bus": 5, "Squeak": 5, "Knock": 5} # 새롭게 클래스 라벨링
    FSD_train_dataset_path = 'D:\\FSDKaggle2018.audio_train'
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
    Urbandataset = pd.read_csv('D:\\UrbanSound8K\\UrbanSound8K\\metadata\\UrbanSound8K.csv')
    Urban_train_dataset_path = 'D:\\UrbanSound8K\\UrbanSound8K\\audio'
    label_setting_UrBan = {"car_horn": 1, "dog_bark": 2, "siren": 4, "street_music": 5, "drilling": 5,
                           "air_conditioner": 5, "jachammer": 5}    # 새롭게 라벨링
    for index, row in Urbandataset.iterrows():
        class_label = row['class']
        if class_label in label_setting_UrBan:  # 우리가 쓸 데이터를 골라내는 작업
            file_name = row['slice_file_name']
            fold_number = row['fold']
            file_path = Urban_train_dataset_path + '\\' + 'fold' + str(fold_number) + '\\' + file_name
            label_check = []        # 나중에 학습할 때 loss계산을 위해 라벨링 형태를 맞춰줌
            for j in range(0, 8):
                if j + 1 == label_setting_UrBan[class_label]:
                    label_check.append(1)
                else:
                    label_check.append(0)
            data, sr = librosa.load(file_path, sr=22050) # librosa 모델을 사용하여 fft
            train_set.append([data, label_check])   # 데이터 저장

    print("데이터 생성 완료")

    return train_set

def AI_HubDataset(train_set):
    Ai_Hub_dataset_path = 'D:\\도시소리'
    Ai_Hub_type_path = ['자동차', '이륜자동차', '동물']
    Ai_Hub_labelset = 'D:\\교통소음'
    label_setting_aihub = {"차량경적": 1, "차량주행음": 5, "차량사이렌": 4, "이륜차경적": 1, "이륜차주행음": 5, "개": 2, "고양이": 3} # 새롭게 라벨링
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
                    for j in range(0, 8):
                        if j + 1 == label_setting_aihub[class_label]:
                            label_check.append(1)
                        else:
                            label_check.append(0)
                    data, sr = librosa.load(data_file_path, sr=22050)   # librosa 모델을 사용하여 fft
                    train_set.append([data, label_check])   # 데이터 저장


    print("데이터 생성 완료")
    return train_set

def AI_HubAlertDataset(train_set):
    AI_HubAlert_dataset_path = 'D:\\경보소리'
    AI_HubAlert_type_path = ["도난경보", "화재경보", "비상경보"]
    AI_HubAlert_label_path = "D:\\경보소리라벨링\\경보"
    label_setting_ai_hubAleart = {"도난경보 소리": 8, "도난 경보음 소리": 8, "침입감지 경보 소리": 8, "화재경보 소리":7, "화재 경보 소리": 7,
                                  "가스누설 화재경보 소리": 7, "자동차 경적 소리": 1, "비상경보 소리": 6, "철도 건널목 신호음 소리": 6, "민방위훈련 사이렌 소리": 6, "공습경보 소리" : 6} # 새롭게 라벨링
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
                for j in range(0, 8):
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
        mfcc = slice(mfcc, 80)     # 설정한 길이에 맞게 맞춰주는 작업

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
            return X, y
        else:
            return X

    def __len__(self):
        return len(self.X)


from torch.utils.data.dataset import random_split
num_epochs = 60 # 학습을 num_epochs만큼 돌림
batch_size = 6   # 배치 사이즈 설정
# 학습을 위한 데이터로 변환(torch.tensor)
test_dataset = Custom_Dataset(X=test_data_X, y=None, train_mode=False)
test_loder = DataLoader(test_dataset, batch_size=6, shuffle=False)
import torch.nn
import torch.nn as nn

# CNN 모델
class CNNclassification(torch.nn.Module): # 4중 layer로 구현
     def __init__(self):
         super(CNNclassification, self).__init__()
         self.layer1 = torch.nn.Sequential(
             nn.Conv2d(120, 200, kernel_size=2, stride=1, padding=1),  # cnn layer
             nn.ReLU(),  # activation function
             nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

         self.layer2 = torch.nn.Sequential(
            nn.Conv2d(200, 200, kernel_size=2, stride=1, padding=1),  # cnn layer
            nn.ReLU(),  # activation function
            nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

         self.layer3 = torch.nn.Sequential(
             nn.Conv2d(200, 200, kernel_size=2, stride=1, padding=1),  # cnn layer
             nn.ReLU(),  # activation function
             nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

         self.layer4 = torch.nn.Sequential(
             nn.Conv2d(200, 200, kernel_size=2, stride=1, padding=1),  # cnn layer
             nn.ReLU(),  # activation function
             nn.MaxPool2d(kernel_size=2, stride=2))  # pooling layer

         self.fc_layer = nn.Sequential(
             nn.Linear(200 * 5 * 1, 8)  # fully connected layer(ouput layer)
         )

     def forward(self, x):

         x = self.layer1(x)
         x = self.layer2(x)
         x = self.layer3(x)
         x = self.layer4(x)
         x = torch.flatten(x, start_dim=1)
         out = self.fc_layer(x)
         return out
from torch.autograd import Variable
class LSTMclassification(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMclassification, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, 8)
        self.relu = nn.ReLU()
    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        output, (hn,cn) = self.lstm(x,(h_0,c_0))
        hn = hn.view(-1,self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

import torch.optim as optim



from tqdm.auto import tqdm

def train(model, optimizer, train_loaders, vail_loaders, scheduler, device, fold):   # 학습
    best_acc = 0
    for epoch in range(1, num_epochs):  # 에포크 설정
        model.train()  # 모델 학습
        running_loss = 0.0
        for wav, label in tqdm(iter(train_loaders)):
            wav, label = wav.to(device).float(), label.to(device).long()  # 배치 데이터
            optimizer.zero_grad()  # 배치마다 optimizer 초기화
            # Data -> Model -> Output
            logit = model(wav)  # 예측값 산출
            loss = criterion(logit, torch.max(label, 1)[1])  # 손실함수 계산
            # 역전파
            loss.backward()  # 손실함수 기준 역전파
            optimizer.step()  # 가중치 최적화
            running_loss += loss.item()

        print('[%d] Train loss: %.10f' % (epoch, running_loss / len(train_loaders)))

        if scheduler is not None:
            scheduler.step()

        # Validation set 평가
        model.eval()  # evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키도록 하는 함수
        vali_loss = 0.0
        correct = 0

        with torch.no_grad():  # 파라미터 업데이트 안하기 때문에 no_grad 사용
            for wav, label in tqdm(iter(vail_loaders)):
                wav, label = wav.to(device).float(), label.to(device).long()
                logit = model(wav)
                vali_loss += criterion(logit, torch.max(label, 1)[1])
                pred = logit.argmax(dim=1, keepdim=True)  # 10개의 class중 가장 값이 높은 것을 예측 label로 추출
                label = label.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()  # 예측값과 실제값이 맞으면 1 아니면 0으로 합산
        vali_acc = 100 * correct / len(vail_loaders.dataset)
        print('Vail set: Loss: {:.4f}, Accuracy: {}/{} ( {:.0f}%)\n'.format(vali_loss / len(vail_loaders), correct,
                                                                            len(vail_loaders.dataset),
                                                                            100 * correct / len(vail_loaders.dataset)))
        acc = 100 * correct / len(vail_loaders.dataset)
        # 베스트 모델 저장
        if best_acc < vali_acc:
            best_acc = vali_acc
            torch.save(model.state_dict(),
                       'C:\\Users\\user\\Desktop\\ai\\best_model' + str(fold) + ".pt")  # 이 디렉토리에 best_model.pth을 저장
            print('Model Saved.')

    return best_acc



test_y = []
for i in train_y:
    for j in range(0, len(i)):
        if i[j] == 1:
            test_y.append(j)
            break

test_y = np.array(test_y)
skf = StratifiedKFold()
skf.get_n_splits(train_X, test_y)
print(skf)
train_X = np.array(train_X)

fold0_model = CNNclassification().to(device)
fold1_model = CNNclassification().to(device)
fold2_model = CNNclassification().to(device)
fold3_model = CNNclassification().to(device)
fold4_model = CNNclassification().to(device)
model_list = [fold0_model, fold1_model, fold2_model, fold3_model, fold4_model]
fold0_lstm_model = LSTMclassification(120, 2, 1).to(device)
fold1_lstm_model = LSTMclassification(120, 2, 1).to(device)
fold2_lstm_model = LSTMclassification(120, 2, 1).to(device)
fold3_lstm_model = LSTMclassification(120, 2, 1).to(device)
fold4_lstm_model = LSTMclassification(120, 2, 1).to(device)
lstm_model_list = [fold0_lstm_model, fold1_lstm_model, fold2_lstm_model, fold3_lstm_model, fold4_lstm_model]
result = {}
for fold, (train_index, vail_index) in enumerate(skf.split(train_X, test_y)):
    print(f"Fold {fold}")
    print('------------------------------------------------')
    train_data_test_X = []
    train_data_test_y = []
    for i in train_index:
        train_data_test_X.append(train_X[i])
        new_label = []
        for label_index in range(0,8):
            if label_index == test_y[i]:
                new_label.append(1)
            else:
                new_label.append(0)
        new_label = np.array(new_label)
        train_data_test_y.append(new_label)

    vail_data_test_X = []
    vail_data_test_y = []
    for i in vail_index:
        vail_data_test_X.append(train_X[i])
        new_label = []
        for label_index in range(0,8):
            if label_index == test_y[i]:
                new_label.append(1)
            else:
                new_label.append(0)
        new_label = np.array(new_label)
        vail_data_test_y.append(new_label)
    print(len(train_index), len(vail_index))
    torch.cuda.empty_cache()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(params=model_list[fold].parameters(), lr=1e-3)
    scheduler = None
    train_data_test_set = Custom_Dataset(train_data_test_X,train_data_test_y)
    train_test_loder = DataLoader(train_data_test_set, batch_size=batch_size, shuffle=True)
    vail_data_test_set = Custom_Dataset(vail_data_test_X, vail_data_test_y)
    vail_test_loder = DataLoader(vail_data_test_set, batch_size=batch_size, shuffle=False)
    acc = train(lstm_model_list[fold], optimizer, train_test_loder, vail_test_loder, scheduler, device, fold)

    result[fold] = acc

    print('Accuracy for fold %d: %d %%' % (fold, acc))
    print('------------------------------------------------')


for key, value in result.items():
    print(f"Fold : {key} ACC : {value}")
def prediction(model, predic_data, device):
    predic_list = []
    model.eval()
    with torch.no_grad():
        for wav in tqdm(iter(predic_data)):
            wav = wav.to(device).float()
            logit = model(wav)
            pred = logit.argmax(dim = 1, keepdim = True)
            predic_list.append(pred.tolist())
    return predic_list

import torch

check_point = torch.load('C:\\Users\\user\\Desktop\\ai\\best_model0.pt', map_location=device)
model = CNNclassification().to(device)
model.load_state_dict(torch.load('C:\\Users\\user\\Desktop\\ai\\best_model0.pt', map_location=device))

data_path = 'E:\\test'
list = []
slice = lambda a, i: a[:, 0:i] if a.shape[1] > i else np.hstack((a, np.zeros((a.shape[0], i - a.shape[1]))))    # 데이터의 길이를 설정한 길이에 맞게 맞춰줌

file_list = os.listdir(data_path)
index = []
for file_name in file_list:
    path = data_path + "\\" + file_name
    data,sr = librosa.load(path, sr=22050)
    test_mfcc = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=40, n_fft=400)
    print(test_mfcc.shape[1], int(test_mfcc.shape[1] / 160))
    test_mfcc_prev = slice(test_mfcc, 80)     # 설정한 길이에 맞게 맞춰주는 작업
    delta_mfcc = librosa.feature.delta(test_mfcc_prev)
    delta_mfcc2 = librosa.feature.delta(test_mfcc_prev, order=2)
    features = np.concatenate([test_mfcc_prev, delta_mfcc, delta_mfcc2], axis=0)
    list.append(features)

list = np.array(list)
list = list.reshape(-1, list.shape[1], list.shape[2], 1)
print(list.shape)
test_da = Custom_Dataset(X=list, y=None, train_mode=False)
test_lod = DataLoader(test_da, batch_size=batch_size, shuffle=False)
preds = prediction(model, test_loder, device)
print(preds)
for i in test_data_y:
    print(i.argmax(), end=" ")
print()
preds = prediction(model, test_lod,device)
print(preds)
