import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
import numpy as np

class LineageOutlierDataset(Dataset):

    def __init__(self, root_dir, train=False, transform=None, aug=1, in_channel_aug = 1):
        self.train = train
        self.transform = transform
        self.data = []
        self.label = []
        self.acc_id = []
        self.aug = aug
        max_cnt = 20000

        if train:
            import random
            #if in_channel_aug != 1:
            cnt = 0
 
            for filename in os.listdir(root_dir):
                file = os.path.join(root_dir,filename)
                with open(file, 'rb') as f:
                    entry = pickle.load(f)
                    if (entry[1][0] != 64 and entry[1][1] != 0):
                        self.data.append(entry[0])
                        self.label.append(entry[1])
                        self.acc_id.append(filename.split('.')[0])

                    else:
                        if cnt < max_cnt:
                            cnt += 1
                            self.data.append(entry[0])
                            self.label.append(entry[1])
                            self.acc_id.append(filename.split('.')[0])

                    if aug != 1:
                        for i in range(aug):
                            idx1 = random.randrange(0,28)
                            idx2 = random.randrange(0,28)
                            entry[0][[idx1,idx2]] = entry[0][[idx2,idx1]]
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

        print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        feature = self.data[idx]
        feature = np.expand_dims(feature.astype('float'),axis=0)

        #print('feature',feature.shape,feature)

        if self.train:
            label = self.label[idx]
            day = label[0]
            price = label[1]
            #print('label',label[0].shape,label[0],label[1])
            #print(feature)
            return feature, day, price, self.acc_id[idx]

        else:
            return feature, self.acc_id[idx]


class LineageDataset(Dataset):

    def __init__(self, root_dir, train=False, transform=None, aug=1, aug_axis_num = 1, in_channel_aug = 1):
        self.train = train
        self.transform = transform
        self.data = []
        self.label = []
        self.acc_id = []
        self.aug = aug

        if train:
            import random
            #if in_channel_aug != 1:
 
            for filename in os.listdir(root_dir):
                file = os.path.join(root_dir,filename)
                with open(file, 'rb') as f:
                    entry = pickle.load(f)
                    self.data.append(entry[0])
                    self.label.append(entry[1])
                    self.acc_id.append(filename.split('.')[0])

                    if aug != 1:
                        for i in range(aug):
                            for i in range(aug_axis_num):
                                idx1 = random.randrange(0,28)
                                idx2 = random.randrange(0,28)
                                entry[0][[idx1,idx2]] = entry[0][[idx2,idx1]]
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
            label = self.label[idx]
            day = label[0]
            price = label[1]
            #print('label',label[0].shape,label[0],label[1])
            #print(feature)
            return feature, day, price, self.acc_id[idx]

        else:
            return feature, self.acc_id[idx]
