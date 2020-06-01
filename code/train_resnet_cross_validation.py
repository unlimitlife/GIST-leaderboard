import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from model.resnet import resnet50, resnet18, resnet34
from datasets import LineageDataset
import os
import numpy as np
from score_function import score_function



#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Hyper-parameters
num_epochs = 60
learning_rate = 0.0005
valid_size = 0.25


"""transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor()])
"""

transforms = transforms.ToTensor()

dataset_train = LineageDataset(root_dir='./data/train', train=True, transform=transforms)


# obtain training indices that will be used for validation
num_train = len(dataset_train)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

#result 
average_accuracy = 0

index = [[indices[:split*(fold-1)] + indices[split*fold:], indices[split*0 : split*1]] for fold in range(1,int(np.floor(1/valid_size))+1)]

for fold, [train_idx, valid_idx] in enumerate(index):

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and smapler)
    train_loader = DataLoader(dataset_train, batch_size=64,
                            sampler = train_sampler, num_workers=4)

    valid_loader = DataLoader(dataset_train, batch_size=64,
                            sampler = valid_sampler, num_workers=4)




    model = resnet34().to(device)

    print(model)


    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay=5e-4)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, [10,25,35,45], gamma=0.2)


    #Train
    total_step = len(train_loader)
    curr_lr = learning_rate

    valid_loss_min = np.Inf
    valid_list = np.empty(0)

    print("===== Train for {}-fold =====".format(fold+1))
    for epoch in range(num_epochs):
        exp_lr_scheduler.step(epoch)

        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for i, (samples, days, prices, acc_id) in enumerate(train_loader):

            #print(samples.shape, days.shape, prices.shape)
            samples = samples.to(device, dtype=torch.float)
            days = days.to(device, dtype=torch.long)
            prices = prices.to(device, dtype=torch.float)

            outputs = model(samples)

            outputs_1 = torch.t(torch.t(outputs)[:][:64])
            outputs_2 = torch.t(torch.t(outputs)[:][64:])

            #print(outputs_1.shape, outputs_2.shape)
            #print(outputs_1, days)

            loss1 = criterion1(outputs_1,days)/100 + 1e-10 
            loss2 = criterion2(outputs_2.squeeze(1),prices) + 1e-10
            loss = loss1 + loss2
            #print(loss1,loss2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*samples.size(0)

            """if (i+1) % 100 == 0:
                print("Epoch [{}/{}], Step[{}/{}] Loss: {:.4f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    """
        model.eval()
        for i, (samples, days, prices, acc_id) in enumerate(valid_loader):

            samples = samples.to(device, dtype=torch.float)
            days = days.to(device, dtype=torch.long)
            prices = prices.to(device, dtype=torch.float)

            outputs = model(samples)

            outputs_1 = torch.t(torch.t(outputs)[:][:64])
            outputs_2 = torch.t(torch.t(outputs)[:][64:])

            loss1 = criterion1(outputs_1,days)/100 + 1e-10
            loss2 = criterion2(outputs_2.squeeze(1),prices) + 1e-10
            loss = loss1 + loss2

            valid_loss += loss.item()*samples.size(0)

        train_loss /= len(train_loader.dataset)
        valid_loss /= len(valid_loader.dataset)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch+1, train_loss, valid_loss))

        valid_list = np.append(valid_list, valid_loss)

        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), 'model_resNet34_fold_'+str(fold+1)+'.pt')
            valid_loss_min = valid_loss




    model.eval()
    day_true = []
    day_predict = []
    price_true = []
    price_predict = []
    acc_id_list = []

    for i, (samples, days, prices, acc_id) in enumerate(valid_loader):
        samples = samples.to(device, dtype=torch.float)
        days = days.to(device, dtype=torch.long)
        prices = prices.to(device, dtype=torch.float)

        outputs = model(samples)

        outputs_1 = torch.t(torch.t(outputs)[:][:64])
        outputs_2 = torch.t(torch.t(outputs)[:][64:])

        _, predicted = torch.max(outputs_1.data, 1)

        day_true.append(days)
        day_predict(predicted)
        price_true.append(prices)
        price_predict.append(outputs_2)
        acc_id_list.append(acc_id)

    predict_label = pd.concat([pd.Series(day_predict_list, index = acc_id_list), pd.Series(price_predict_list, index = acc_id_list)], axis=1)
    actual_label = pd.concat([pd.Series(day_true_list, index = acc_id_list), pd.Series(price_true_list, index = acc_id_list)], axis=1)

    predict_label.to_csv('predict_label.csv')
    actual_label.to_csv('actual_label.csv')

    print('++SCORE++',score_function('predict_label.csv','actual_label.csv'))


