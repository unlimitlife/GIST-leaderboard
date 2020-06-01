import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from model.binary_resnet import resnet50,resnet34,resnet18
from datasets import LineageDataset
from score_function import score_function
import os
import numpy as np
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = transforms.ToTensor()
dataset_test1 = LineageDataset(root_dir='./data/test1', train=False, transform=transforms)
dataset_test2 = LineageDataset(root_dir='./data/test2', train=False, transform=transforms)

test1_loader = DataLoader(dataset_test1, batch_size=128,num_workers=2)
test2_loader = DataLoader(dataset_test2, batch_size=128,num_workers=2)



#model = resnet50()
#model.load_state_dict(torch.load('model_resNet50.pt'))
#model = seresnet50()
#model.load_state_dict(torch.load('model_seresNet50.pt'))
model = resnet34()
model.load_state_dict(torch.load('model_resnet34_best_score_custom_loss2_10510.pt'))
model = nn.DataParallel(model)
#model.load_state_dict(torch.load('model_resNet34_best_score_8158_aug.pt'))
model.to(device)
print(model)

with torch.no_grad():
    model.eval()

    day_predict = []
    price_predict = []
    acc_id_list = []

    for i, (samples, acc_id) in enumerate(test1_loader):
        samples = samples.to(device, dtype=torch.float)

        outputs = model(samples)

        #outputs_1 = torch.round(torch.clamp(outputs[:, :1], 1, 64))
        #outputs_2 = torch.clamp(outputs[:, 1:], 0)

        outputs_1 = torch.round(torch.clamp(torch.t(torch.t(outputs)[:][:1]), 1, 64))
        outputs_2 = torch.clamp(torch.t(torch.t(outputs)[:][1:]), 0)

        day_predict.extend(torch.flatten(outputs_1).data.cpu().numpy())
        price_predict.extend(torch.flatten(outputs_2).data.cpu().numpy())
        acc_id_list.extend(acc_id)

    idx = range(1,10001)


    #predict_label = pd.concat([pd.Series(day_predict, index = acc_id_list), pd.Series(price_predict, index = acc_id_list)], axis=1)
    #actual_label = pd.concat([pd.Series(day_true, index = acc_id_list), pd.Series(price_true, index = acc_id_list)], axis=1)

    predict_label = pd.concat([pd.concat([pd.DataFrame(acc_id_list,columns=['acc_id']),pd.DataFrame(day_predict,columns=['survival_time'])],axis=1), pd.DataFrame(price_predict,columns=['amount_spent'])],axis=1)

    predict_label.to_csv('test1_predict.csv', index=False)


    model.eval()

    day_predict = []
    price_predict = []
    acc_id_list = []

    for i, (samples, acc_id) in enumerate(test2_loader):
        samples = samples.to(device, dtype=torch.float)

        outputs = model(samples)

        outputs_1 = torch.round(torch.clamp(outputs[:, :1], 1, 64))
        outputs_2 = torch.clamp(outputs[:, 1:], 0)

        _, predicted = torch.max(outputs_1.data, 1)
        predicted += 1

        day_predict.extend(torch.flatten(predicted).data.cpu().numpy())
        price_predict.extend(torch.flatten(outputs_2).data.cpu().numpy())
        acc_id_list.extend(acc_id)

    idx = range(1,10001)



    predict_label = pd.concat([pd.concat([pd.DataFrame(acc_id_list,columns=['acc_id']),pd.DataFrame(day_predict,columns=['survival_time'])],axis=1), pd.DataFrame(price_predict,columns=['amount_spent'])],axis=1)

    predict_label.to_csv('test2_predict.csv', index=False)



