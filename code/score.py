import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from model.binary_resnet import resnet50,resnet34,resnet18
#from model.custom_resnet import resnet34
from datasets import LineageDataset
from score_function import score_function
import os
import numpy as np
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = transforms.ToTensor()
dataset_val = LineageDataset(root_dir='./data/val', train=True, transform=transforms)
val_loader = DataLoader(dataset_val, batch_size=128,num_workers=2)



#model = resnet50()
#model.load_state_dict(torch.load('model_resNet50.pt'))
#model = seresnet50()
#model.load_state_dict(torch.load('model_seresNet50.pt'))
model = resnet34()
#model.load_state_dict(torch.load('model_resNet34_best_score_8158_aug.pt'))
model.load_state_dict(torch.load('model_resnet34_best_score_custom_loss2_10510.pt'))

model = nn.DataParallel(model)
#model.load_state_dict(torch.load('model_resNet50_best_score_7900.pt'))
#model.load_state_dict(torch.load('model_resNet34_best_valid_7800.pt'))
model.to(device)
print(model)

with torch.no_grad():
    model.eval()

    day_true = []
    day_predict = []
    price_true = []
    price_predict = []
    acc_id_list = []

    for i, (samples, days, prices, acc_id) in enumerate(val_loader):

        samples = samples.to(device, dtype=torch.float)

        outputs = model(samples)

        outputs_1 = torch.round(torch.clamp(outputs[:, :1], 1, 64))
        outputs_2 = torch.clamp(outputs[:, 1:], 0)
        #outputs_2 = outputs[:, 1:]


        day_predict.extend(torch.flatten(outputs_1).data.cpu().numpy())
        price_predict.extend(torch.flatten(outputs_2).data.cpu().numpy())
        acc_id_list.extend(acc_id)

        day_true.extend(torch.flatten(days).data.cpu().numpy())
        price_true.extend(torch.flatten(prices).data.cpu().numpy())

    idx = range(1,10001)
    print('day_true',day_true)
    print('day_predict',day_predict)
    print('price_true',price_true)
    print('price_predict',price_predict)
    print('acc_id_list',acc_id_list)


    #predict_label = pd.concat([pd.Series(day_predict, index = acc_id_list), pd.Series(price_predict, index = acc_id_list)], axis=1)
    #actual_label = pd.concat([pd.Series(day_true, index = acc_id_list), pd.Series(price_true, index = acc_id_list)], axis=1)

    predict_label = pd.concat([pd.concat([pd.DataFrame(acc_id_list,columns=['acc_id']),pd.DataFrame(day_predict,columns=['survival_time'])],axis=1), pd.DataFrame(price_predict,columns=['amount_spent'])],axis=1)
    actual_label = pd.concat([pd.concat([pd.DataFrame(acc_id_list,columns=['acc_id']),pd.DataFrame(day_true,columns=['survival_time'])],axis=1), pd.DataFrame(price_true,columns=['amount_spent'])],axis=1)
    print(predict_label)
    print(actual_label)

    #predict_label.to_csv('val_resnet34_8158_aug_predict_no_clamp.csv', index=False)
    predict_label.to_csv('val_resnet34_best.csv', index=False)
    actual_label.to_csv('actual_label.csv', index=False)

    print('++SCORE++',score_function('val_resnet34_best.csv','actual_label.csv'))


