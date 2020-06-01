import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from model.resnet import resnet50,resnet34
#from model.custom_resnet import resnet50,resnet34,resnet18
from datasets import LineageDataset
import os
import numpy as np




#Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Hyper-parameters
num_epochs = 80
learning_rate = 0.01


"""transforms = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor()])
"""

transforms = transforms.ToTensor()

dataset_train = LineageDataset(root_dir='./data/train', train=True, transform=transforms)


# prepare data loaders (combine dataset and smapler)
train_loader = DataLoader(dataset_train, batch_size=128,
                         num_workers=4)

valid_loader = DataLoader(dataset_train, batch_size=128,
                         num_workers=4)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


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

model = nn.DataParallel(resnet34()).to(device)

print(model)


#criterion1 = nn.CrossEntropyLoss()
criterion1 = FocalLoss()
criterion2 = nn.SmoothL1Loss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9, weight_decay=5e-4)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, [20,45,60], gamma=0.2)


#Train
total_step = len(train_loader)
curr_lr = learning_rate

valid_loss_min = np.Inf

for epoch in range(num_epochs):
    exp_lr_scheduler.step(epoch)

    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for i, (samples, days, prices, acc_id) in enumerate(train_loader):

        samples = samples.to(device, dtype=torch.float)
        days = days.to(device, dtype=torch.long)
        prices = prices.to(device, dtype=torch.float)

        outputs = model(samples)

        outputs_1 = torch.t(torch.t(outputs)[:][:64])
        outputs_2 = torch.clamp(torch.t(torch.t(outputs)[:][64:]) ,0)

        print(outputs_1, days)
        print(outputs_1.shape, days.shape)

        loss1 = criterion1(outputs_1,days) + 1e-10
        loss2 = criterion2(outputs_2.squeeze(1),prices) + 1e-10
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*samples.size(0)

    model.eval()
    for i, (samples, days, prices, acc_id) in enumerate(valid_loader):

        samples = samples.to(device, dtype=torch.float)
        days = days.to(device, dtype=torch.long)
        prices = prices.to(device, dtype=torch.float)

        outputs = model(samples)

        outputs_1 = torch.t(torch.t(outputs)[:][:64])
        outputs_2 = torch.t(torch.t(outputs)[:][64:])

        loss1 = criterion1(outputs_1,days) + 1e-10
        loss2 = criterion2(outputs_2.squeeze(1),prices) + 1e-10
        loss = loss1 + loss2

        valid_loss += loss.item()*samples.size(0)

    train_loss /= len(train_loader.dataset)
    valid_loss /= len(valid_loader.dataset)

    # print training/validation statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch+1, train_loss, valid_loss))


    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        save_module = list(model.children())[0]
        torch.save(save_module.state_dict(), 'model_resNet34.pt')
        valid_loss_min = valid_loss
