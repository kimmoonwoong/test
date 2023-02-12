import Custom_Dataset
import data_setting
from torch.utils.data import DataLoader, Dataset

num_epochs = 100

batch_size = 10

train_dataset = Custom_Dataset(X=data_setting.train_X, y = data_setting.train_y)
train_loader =DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
