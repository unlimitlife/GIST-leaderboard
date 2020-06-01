import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from model.binary_resnet import resnet50,resnet34
from model.seresnet import seresnet50,seresnet34
from model.efficientnet import *
#from model.custom_resnet import resnet50,resnet34,resnet18
from datasets import LineageDataset, LineageOutlierDataset
import os
import numpy as np
#from loss_og import ScoreLoss
from loss_custom import ScoreLoss
import pandas as pd
from score_function import score_function


#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Hyper-parameters
num_epochs = 100
learning_rate = 0.01
aug_num = 1


transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor()])

####other aug2
"""transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor()])
"""
#transforms = transforms.ToTensor()


dataset_train = LineageDataset(root_dir='./data/train', train=True, transform=transforms, aug=aug_num, aug_axis_num=1)
#dataset_train = LineageOutlierDataset(root_dir='./data/train', train=True, transform=transforms, aug=aug_num)


# prepare data loaders (combine dataset and smapler)
train_loader = DataLoader(dataset_train, batch_size=128,
                         num_workers=4, shuffle=True)

dataset_val = LineageDataset(root_dir='./data/val', train=True, transform=transforms)

val_loader = DataLoader(dataset_val, batch_size=128,num_workers=2)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform(m.weight.data)
        #torch.nn.init.xavier_uniform(m.bias.data)


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

model = resnet34()
#model = resnet50()
#model.apply(init_weights)

#model = efficientnet_b4(num_classes=2)
model = nn.DataParallel(model).to(device)
print(model)


criterion1 = ScoreLoss()
#criterion1 = FocalLoss()
#criterion2 = nn.SmoothL1Loss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay=5e-4)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, [12,25,35,45], gamma=0.2)


#Train
total_step = len(train_loader)
curr_lr = learning_rate

valid_loss_min = np.Inf
score_max = 0

for epoch in range(num_epochs):
    exp_lr_scheduler.step(epoch)

    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for i, (samples, days, prices, acc_id) in enumerate(train_loader):

        samples = samples.to(device, dtype=torch.float)
        days = days.to(device, dtype=torch.float)
        prices = prices.to(device, dtype=torch.float)

        outputs = model(samples)
        #outputs_copy = torch.empty_like(outputs).copy_(outputs)

        #outputs_1 = torch.clamp(outputs[:, :1], 1, 64).to(device, dtype=torch.float)
        #outputs_2 = torch.clamp(outputs[:, 1:], 0)

        #outputs_1 = outputs_1.squeeze(1)
        #print(outputs_1.shape, days.shape)


        loss = criterion1(outputs, days, prices)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*samples.size(0)

    model.eval()
    for i, (samples, days, prices, acc_id) in enumerate(val_loader):

        samples = samples.to(device, dtype=torch.float)
        days = days.to(device, dtype=torch.float)
        prices = prices.to(device, dtype=torch.float)

        outputs = model(samples)

        loss = criterion1(outputs, days, prices)

        valid_loss += loss.item()*samples.size(0)

    train_loss /= len(train_loader.dataset)
    valid_loss /= len(val_loader.dataset)

    model.eval()

    day_true = []
    day_predict = []
    price_true = []
    price_predict = []
    acc_id_list = []

    for i, (samples, days, prices, acc_id) in enumerate(val_loader):
        samples = samples.to(device, dtype=torch.float)
        days = days.to(device, dtype=torch.float)
        prices = prices.to(device, dtype=torch.float)

        outputs = model(samples)

        outputs_1 = torch.round(torch.clamp(torch.t(torch.t(outputs)[:][:1]), 1, 64))
        outputs_2 = torch.clamp(torch.t(torch.t(outputs)[:][1:]), 0)

        day_true.extend(torch.flatten(days).data.cpu().numpy())
        day_predict.extend(torch.flatten(outputs_1).data.cpu().numpy())
        price_true.extend(torch.flatten(prices).data.cpu().numpy())
        price_predict.extend(torch.flatten(outputs_2).data.cpu().numpy())
        acc_id_list.extend(acc_id)

    idx = range(1,10001)

    predict_label = pd.concat([pd.concat([pd.DataFrame(acc_id_list,columns=['acc_id']),pd.DataFrame(day_predict,columns=['survival_time'])],axis=1), pd.DataFrame(price_predict,columns=['amount_spent'])],axis=1)
    actual_label = pd.concat([pd.concat([pd.DataFrame(acc_id_list,columns=['acc_id']),pd.DataFrame(day_true,columns=['survival_time'])],axis=1), pd.DataFrame(price_true,columns=['amount_spent'])],axis=1)

    predict_label.to_csv('val_predict.csv', index=False)
    actual_label.to_csv('val2_predict.csv', index=False)

    score = score_function('val_predict.csv','val2_predict.csv')
    print('++SCORE++', score)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch+1, train_loss, valid_loss))


    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        save_module = list(model.children())[0]
        torch.save(save_module.state_dict(), 'model_resnet50_best_valid_hanta.pt')
        valid_loss_min = valid_loss

    if score >= score_max:
        print('New score RECORD !! ({:.6f} --> {:.6f}).  Saving model ...'.format(
        score_max,
        score))
        save_module = list(model.children())[0]
        torch.save(save_module.state_dict(), 'model_resnet50_best_score_hanta.pt')
        score_max = score

