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
from binary_datasets import LineageDataset
from score_function import score_function
import os
import numpy as np
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = transforms.ToTensor()
dataset_val_d = LineageDataset(root_dir='./data_binary/day/val', train=True, transform=transforms)
val_loader_d = DataLoader(dataset_val_d, batch_size=32,num_workers=2)

dataset_val_p = LineageDataset(root_dir='./data_binary/price/val', train=True, transform=transforms)
val_loader_p = DataLoader(dataset_val_p, batch_size=32,num_workers=2)

dataset_val = LineageDataset(root_dir='./data/test1', train=True, transform=transforms)
val_loader = DataLoader(dataset_val, batch_size=32,num_workers=2)


#model = resnet50()
#model.load_state_dict(torch.load('model_resNet50.pt'))
#model = seresnet50()
#model.load_state_dict(torch.load('model_seresNet50.pt'))
model_d = resnet34()
model_p = resnet34()
#model = nn.DataParallel(model)
#model.load_state_dict(torch.load('model_resNet34_best_score_8158_aug.pt'))
print(model_d)
model_d.load_state_dict(torch.load('model_resNet34_bin_day.pt'))
model_p.load_state_dict(torch.load('model_resNet34_bin_price.pt'))

predict_label = pd.read_csv('test1_predict.csv')
#model.load_state_dict(torch.load('model_resNet50_best_score_7900.pt'))
#model.load_state_dict(torch.load('model_resNet34_best_valid_7800.pt'))
model_d.to(device)
model_p.to(device)
acc_id_list = predict_label['acc_id'].copy().tolist()
print(acc_id_list)
def mergeLeftInOrder(x, y, on=None):
    x = x.copy()
    x["Order"] = np.arange(len(x))
    z = x.merge(y, how='left', on=on).set_index("Order").ix[np.arange(len(x)), :]
    return z


with torch.no_grad():
    model_d.eval()
    model_p.eval()

    day_true = []
    day_predict = []
    price_true = []
    price_predict = []
    #acc_id_list = []
    acc_id_list_d = []
    acc_id_list_p = []
    day_predict_dict = {}
    price_predict_dict = {}

    #for i, (samples, days, prices, acc_id) in enumerate(val_loader):

        #acc_id_list.extend(acc_id)
        #day_true.extend(torch.flatten(days).data.cpu().numpy())
        #price_true.extend(torch.flatten(prices).data.cpu().numpy())

    for i, (samples, acc_id) in enumerate(val_loader):

        samples = samples.to(device, dtype=torch.float)

        outputs = model_d(samples)
        _, predicted = torch.max(outputs.data, 1)

        #day_predict.extend(torch.flatten(outputs).data.cpu().numpy())
        day_predict.extend(torch.flatten(predicted).data.cpu().numpy())
        acc_id_list_d.extend(list(map(int, acc_id)))
        #acc_id_list_d.extend(acc_id)
        #for acc in acc_id:
            #print(acc)
            #print(acc in acc_id_list)

    for i, acc_id in enumerate(acc_id_list_d):
        #print(acc_id)
        day_predict_dict[acc_id_list.index(acc_id)] = (acc_id,day_predict[i])

    for i, (samples, acc_id) in enumerate(val_loader):

        samples = samples.to(device, dtype=torch.float)

        outputs = model_p(samples)
        _, predicted = torch.max(outputs.data, 1)

        #price_predict.extend(torch.flatten(outputs).data.cpu().numpy())
        price_predict.extend(torch.flatten(predicted).data.cpu().numpy())
        acc_id_list_p.extend(list(map(int, acc_id)))

    for i, acc_id in enumerate(acc_id_list_p):
        price_predict_dict[acc_id_list.index(acc_id)] = (acc_id,price_predict[i])

    #predict_label = pd.concat([pd.Series(day_predict, index = acc_id_list), pd.Series(price_predict, index = acc_id_list)], axis=1)
    #actual_label = pd.concat([pd.Series(day_true, index = acc_id_list), pd.Series(price_true, index = acc_id_list)], axis=1)


    #predict_label_d = pd.concat([pd.DataFrame(acc_id_list_d,columns=['acc_id']),pd.DataFrame(day_predict,columns=['survival_time_b'])],axis=1)
    #predict_label_p = pd.concat([pd.DataFrame(acc_id_list_p,columns=['acc_id']),pd.DataFrame(price_predict,columns=['amount_spent_b'])],axis=1)

    #predict_label_d.set_index('acc_id',inplace=True)
    #predict_label_p.set_index('acc_id',inplace=True)

    #predict_label.acc_id = predict_label.acc_id.astype(int)
    #predict_label_d.acc_id = predict_label_d.acc_id.astype(int)
    accid_list = []
    day_list = []
    for day_predict_key in day_predict_dict.keys():
        (acc_id, day) = day_predict_dict[day_predict_key]
        accid_list.append(acc_id)
        day_list.append(day)

    day_predict_dict = {}
    day_predict_dict['acc_id'] = accid_list
    day_predict_dict['survival_time_b'] = day_list

        
    accid_list = []
    price_list = []
    for price_predict_key in price_predict_dict.keys():
        (acc_id, price) = price_predict_dict[price_predict_key]
        accid_list.append(acc_id)
        price_list.append(price)

    price_predict_dict = {}
    price_predict_dict['acc_id'] = accid_list
    price_predict_dict['amount_spent_b'] = day_list
    #predict_label_d = pd.DataFrame(list(day_predict_dict.items()),columns=['acc_id','survival_time_b'])
    #predict_label_p = pd.DataFrame(list(day_predict_dict.items()),columns=['acc_id','survival_time_b'])
    
    predict_label_d = pd.DataFrame.from_dict(day_predict_dict)
    #predict_label_d.columns = ['acc_id','survival_time_b']
    predict_label_p = pd.DataFrame.from_dict(price_predict_dict)
    #predict_label_p.columns = ['acc_id','amount_spent_b']
    
    print(predict_label_d)
    print(predict_label_p)
    print("d",type(predict_label_d['acc_id'][3]))
    print("d",type(predict_label_d['survival_time_b'][3]))
    print("d",type(predict_label_d))
    #print("p",predict_label_p)
    print("",type(predict_label['acc_id'][3]))
    print("",type(predict_label['survival_time'][3]))
    print("",type(predict_label.amount_spent))
    predict_label_d.to_csv('asd.csv', index=False)
    predict_label = pd.merge(predict_label, predict_label_d, how='left',on=['acc_id'])
    #predict_label = mergeLeftInOrder(predict_label,predict_label_d,on='acc_id')
    print("",predict_label)
    predict_label.loc[predict_label['survival_time_b']==1,['survival_time']]=64
    predict_label = predict_label.drop('survival_time_b',axis=1)

    print("",predict_label)
    predict_label = pd.merge(predict_label, predict_label_p, how='left', on=['acc_id'])
    print("",predict_label)
    #predict_label = mergeLeftInOrder(predict_label,predict_label_p,on='acc_id')
    predict_label.loc[predict_label['amount_spent_b']==1,['amount_spent']]=0

    predict_label = predict_label.drop('amount_spent_b',axis=1)
    print("",predict_label)

    #actual_label = pd.concat([pd.concat([pd.DataFrame(acc_id_list,columns=['acc_id']),pd.DataFrame(day_true,columns=['survival_time'])],axis=1), pd.DataFrame(price_true,columns=['amount_spent'])],axis=1)
    print(predict_label)
    #print(actual_label)

    #predict_label.to_csv('val_resnet34_8158_aug_predict_no_clamp.csv', index=False)
    predict_label.to_csv('test1_predict_bin.csv', index=False)
    #actual_label.to_csv('actual_label.csv', index=False)

    #print('++SCORE++',score_function('val_resnet34_best_w_bin.csv','actual_label.csv'))


