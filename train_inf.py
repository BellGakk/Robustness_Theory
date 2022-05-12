from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime
import numpy as np

from networks.single_hidden import *
from torch.autograd import Variable
from utils import * 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='./log/')

parser = argparse.ArgumentParser(description='Single Hidden Layer Training.')
parser.add_argument('--magnitude', default=1.0, type=float)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--log_intervals', default=10, type=int)
parser.add_argument('--classes', default=2, type=int)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--optimizer', default='sgd', type=str)
parser.add_argument('--loss', default='mse', type=str)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--batch_size', default=10000, type=int)
parser.add_argument('--net_type', default='mlp', type=str)
parser.add_argument('--depth', default=1, type=int)
parser.add_argument('--training_type', default='std', type=str)
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--bn', default=False, type=bool)

args = parser.parse_args()
use_cuda = torch.cuda.is_available()
params = dict()
best_acc = 0.0
if args.loss == 'ce':
    criterion = nn.CrossEntropyLoss()
elif args.loss == 'mse':
    criterion = nn.MSELoss()


#Data Loading
print('\n [Phase 1]: Data Loading')

transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

if (args.dataset == 'cifar10'):
    params['class_size'] = 5000
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    
    X_traindata = torch.zeros(5000*args.classes, 32*32*3)
    if args.loss == 'ce':
        Y_traindata = torch.zeros(5000*args.classes)
    elif args.loss == 'mse':
        Y_traindata = torch.zeros(5000*args.classes, args.classes)
    count = 0
    for index in range(len(trainset)):
        data, label = trainset[index]
        if label == 0 or label == 1:
            data = data.view(-1)
            input_len = len(data)
            X_traindata[count] = data
            Y_traindata[count] = torch.zeros(2)
            Y_traindata[count][label] += 1
            count += 1
    
    X_testdata = torch.zeros(1000*args.classes, 32*32*3)
    if args.loss == 'ce':
        Y_testdata = torch.zeros(1000*args.classes, 32*32*3)
    elif args.loss == 'mse':
        Y_testdata = torch.zeros(1000*args.classes, args.classes)
    count = 0
    for index in range(len(testset)):
        data, label = testset[index]
        if label == 0 or label == 1:
            data = data.view(-1)                
            X_testdata[count] = data
            Y_testdata[count] = torch.ones(2)
            Y_testdata[count][label] += 1
            count += 1
    
    specific_trainset = GetDataset(X_traindata, Y_traindata)
    specific_testset = GetDataset(X_testdata, Y_testdata)
    params['input_size'] = len(specific_trainset[0][0])

elif(args.dataset == 'mnist'):
    print("| Preparing mnist dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform_test)
    
trainloader = torch.utils.data.DataLoader(specific_trainset, batch_size=5000*args.classes, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(specific_testset, batch_size=1000*args.classes, shuffle=True, num_workers=2)

print('\n [Phase 2]: Model Setup')
mlp, file_name = getNetwork(args, params)
for module in mlp.modules():
    print(module)
mlp.apply(param_init)

if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
dataset_point = './checkpoint/'+args.dataset
if not os.path.isdir(dataset_point):
    os.mkdir(dataset_point)
training_type_point = dataset_point + os.sep + args.training_type
if not os.path.isdir(training_type_point):
    os.mkdir(training_type_point)
model_type_point = training_type_point + os.sep + file_name
if not os.path.isdir(model_type_point):
    os.mkdir(model_type_point)


def train(epoch):
    mlp.train()
    mlp.training = True
    train_loss = 0.0
    correct = 0
    total = 0

    optimizer = optim.SGD(mlp.parameters(), lr=cf.learning_rate(epoch, args), momentum=args.momentum, weight_decay=args.weight_decay)
    print('\n => Training Epoch #%d, LR=%.4f ' % (epoch, args.lr))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.loss == 'ce':
            targets = targets.type(torch.LongTensor)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = mlp(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            if args.loss == 'mse':
                _, predicted_targets = torch.max(targets.data, 1)
                correct += predicted.eq(predicted_targets).cpu().sum()
            elif args.loss == 'ce':
                correct += predicted.eq(targets.data).cpu().sum()
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %(epoch, args.num_epochs, batch_idx+1,(len(specific_trainset)//args.batch_size), loss.item(), 100*correct/total))
            sys.stdout.flush()
    writer.add_scalars(main_tag='TrainLoss', tag_scalar_dict={file_name:train_loss}, global_step=epoch)
    writer.add_scalars(main_tag='TrainAcc', tag_scalar_dict={file_name:(correct/total)}, global_step=epoch)


def test(epoch):
    global best_acc
    mlp.eval()
    mlp.training=False
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if args.loss == 'ce':
                targets = targets.type(torch.LongTensor)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = mlp(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            if args.loss == 'ce':
                correct += predicted.eq(targets.data).cpu().sum()
            elif args.loss == 'mse':
                _, predicted_targets = torch.max(targets.data, 1)
                correct += predicted.eq(predicted_targets).cpu().sum()

        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))

        if acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
            state = {
                    'net':mlp.state_dict(),
                    'acc':acc,
                    'epoch':epoch,
            }
            torch.save(state, model_type_point+os.sep+file_name+'-epoch'+str(epoch)+'.pt')
            best_acc = acc
    writer.add_scalars(main_tag='TestLoss', tag_scalar_dict={file_name:test_loss}, global_step=epoch)
    writer.add_scalars(main_tag='TestAcc', tag_scalar_dict={file_name:(correct/total)}, global_step=epoch)

if __name__ == '__main__':
    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(args.num_epochs))
    print('| Initial Learning Rate = ' + str(args.lr))
    print('| Optimizer = ' + args.optimizer)
    print('| Loss Function = ' + args.loss)
    
    mlp.cuda()

    elapsed_time = 0.0
    for epoch in range(args.start_epoch, args.start_epoch+args.num_epochs):
        start_time = time.time()
        if args.training_type == 'std':
            train(epoch)
            test(epoch)
        epoch_time = time.time() - start_time
        print('| Epoch time : %d:%02d:%02d'  %(cf.get_hms(epoch_time)))
        elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))
    
    

    



