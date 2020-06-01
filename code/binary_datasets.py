import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
import numpy as np

class LineageDataset(Dataset):

    def __init__(self, root_dir, train=False, transform=None):
        self.train = train
        self.transform = transform
        self.data = []
        self.label = []
        self.acc_id = []


        if train:
            for filename in os.listdir(root_dir):
                file = os.path.join(root_dir,filename)
                with open(file, 'rb') as f:
                    entry = pickle.load(f)
                    self.data.append(entry[0])
                    self.label.append(entry[1])
                    self.acc_id.append(filename.split('.')[0])

        else:
            for filename in os.listdir(root_dir):
                file = os.path.join(root_dir,filename)
                with open(file, 'rb') as f:
                    entry = pickle.load(f)
                    self.data.append(entry)
                    self.acc_id.append(filename.split('.')[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        feature = self.data[idx]
        feature = np.expand_dims(feature.astype('float'),axis=0)

        #print('feature',feature.shape,feature)

        if self.train:
            #label_idx = self.label[idx]

            #label = np.zeros((1,2))
            #label[0][label_idx] = 1
            #return feature, label, self.acc_id[idx]
            return feature, self.label[idx][0], self.acc_id[idx]

        else:
            return feature, self.acc_id[idx]
