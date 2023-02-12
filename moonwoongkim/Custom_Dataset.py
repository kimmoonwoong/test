import torchvision.datasets as datasets
import torchvision.transforms as transforms
import data_setting
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset

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
