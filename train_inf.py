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
import csv

from attacks.whitebox import *
from networks.mlp import *
from torch.autograd import Variable
from utils import * 
from dataset import * 
from torch.utils.tensorboard import SummaryWriter
tensorboard_writer = SummaryWriter(log_dir='../logs_MLP/')

parser = argparse.ArgumentParser(description='Single Hidden Layer Training.')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--log_intervals', default=10, type=int)
parser.add_argument('--classes', default=10, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--loss', default='ce', type=str)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--net_type', default='mlp', type=str)
parser.add_argument('--depth', default=1, type=int)
parser.add_argument('--training_type', default='std', type=str)
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--bn', default=False, type=bool)
parser.add_argument('--widen_factor', default=1, type=int)
parser.add_argument('--gpu', default=0, type=int)

args = parser.parse_args()
if torch.cuda.is_available():
    device = torch.device('cuda:'+str(args.gpu))
else:
    device = torch.device('cpu')
    print('The GPUs are currently not available, please use cpu.')
params = dict()
best_acc = 0.0



#Data Loading
print('\n [Phase 1]: Data Loading')

if (args.dataset == 'cifar10'):
    params['train_size'] = 5000
    params['test_size'] = 1000
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset, testset = cifar10()

elif(args.dataset) == 'cifar100':
    params['train_size'] = 500
    params['test_size'] = 100
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset, testset = cifar100()

elif(args.dataset == 'mnist'):
    params['train_size'] = 5000
    params['test_size'] = 1000
    print("| Preparing mnist dataset...")
    sys.stdout.write("| ")
    trainset, testset = mnist()

if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    X_traindata = torch.zeros(params['train_size']*args.classes, 32*32*3)
    X_testdata = torch.zeros(params['test_size']*args.classes, 32*32*3)
elif args.dataset == 'mnist':
    X_traindata = torch.zeros(params['train_size']*args.classes, 28*28)
    X_testdata = torch.zeros(params['test_size']*args.classes, 28*28)
if args.loss == 'ce':
    Y_traindata = torch.zeros(params['train_size']*args.classes)
    Y_testdata = torch.zeros(params['test_size']*args.classes)
elif args.loss == 'mse':
    Y_traindata = torch.zeros(params['train_size']*args.classes, args.classes)
    Y_testdata = torch.zeros(params['test_size']*args.classes, args.classes)

count = 0
candidate_classes = [i for i in range(args.classes)]
for index in range(len(trainset)):
    data, label = trainset[index]
    if label in candidate_classes:
        data = data.view(-1)
        X_traindata[count] = data
        if args.loss == 'mse':
            Y_traindata[count][label] += 1
        elif args.loss == 'ce':
            Y_traindata[count] = label
        count += 1
count = 0 
for index in range(len(testset)):
    data, label = testset[index]
    if label in candidate_classes:
        data = data.view(-1)                
        X_testdata[count] = data
        if args.loss == 'mse':
            Y_testdata[count][label] += 1
        elif args.loss == 'ce':
            Y_testdata[count] = label
        count += 1
    
specific_trainset = GetDataset(X_traindata, Y_traindata)
specific_testset = GetDataset(X_testdata, Y_testdata)

params['input_size'] = len(specific_trainset[0][0])
    
trainloader = torch.utils.data.DataLoader(specific_trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(specific_testset, batch_size=args.batch_size, shuffle=True, num_workers=2)

print('\n [Phase 2]: Model Setup')
mlp, file_name = getNetwork(args, params)
for module in mlp.modules():
    print(module)
#mlp.apply(param_init)

if not os.path.isdir('../checkpoint'):
    os.mkdir('../checkpoint')
dataset_point = '../checkpoint/'+args.dataset
if not os.path.isdir(dataset_point):
    os.mkdir(dataset_point)
training_type_point = dataset_point + os.sep + args.training_type
if not os.path.isdir(training_type_point):
    os.mkdir(training_type_point)
model_type_point = training_type_point + os.sep + file_name
if not os.path.isdir(model_type_point):
    os.mkdir(model_type_point)
if not os.path.isdir('../csvs'):
    os.mkdir('../csvs')
csv_save_dir = os.path.join('../csvs', file_name)
if not os.path.isdir(os.path.join(csv_save_dir)):
    os.mkdir(csv_save_dir)


def train(epoch, optimizer, scheduler):
    mlp.train()
    mlp.training = True
    train_loss = 0.0
    correct = 0
    total = 0
    
    print('\n => Training Epoch #%d, LR=%.4f ' % (epoch, args.lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.loss == 'ce':
            targets = targets.type(torch.LongTensor)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        if args.training_type == 'std':
            outputs = mlp(inputs)
        elif args.training_type == 'fgsm':
            inputs_adv = attack.perturb(inputs, targets)
            outputs = mlp(inputs_adv)
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
        
        scheduler.step()
        acc = (correct / total).item()
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %(epoch, args.num_epochs, batch_idx+1,(len(specific_trainset)//args.batch_size)+1, loss.item(), 100*acc))
        sys.stdout.flush()
    
    tensorboard_writer.add_scalars(main_tag='TrainLoss', tag_scalar_dict={file_name:train_loss/args.batch_size}, global_step=epoch)
    tensorboard_writer.add_scalars(main_tag='TrainAcc', tag_scalar_dict={file_name:acc}, global_step=epoch)
    return train_loss/args.batch_size , acc

def test(epoch):
    global best_acc
    mlp.eval()
    mlp.training=False
    test_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if args.loss == 'ce':
            targets = targets.type(torch.LongTensor)
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs), Variable(targets)
        if args.training_type == 'std':
            outputs = mlp(inputs)
        elif args.training_type == 'fgsm':
            inputs_adv = attack.perturb(inputs, targets)
            outputs = mlp(inputs_adv)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        if args.loss == 'ce':
            correct += predicted.eq(targets.data).cpu().sum()
        elif args.loss == 'mse':
            _, predicted_targets = torch.max(targets.data, 1)
            correct += predicted.eq(predicted_targets).cpu().sum()

    acc = (correct/total).item()
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, test_loss, 100*acc))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        state = {
                'net':mlp.state_dict(),
                'acc':acc,
                'epoch':epoch,
        }
    torch.save(state, model_type_point+os.sep+file_name+'-epoch'+str(epoch)+'.pt')
    best_acc = acc
    tensorboard_writer.add_scalars(main_tag='TestLoss', tag_scalar_dict={file_name:test_loss/args.batch_size}, global_step=epoch)
    tensorboard_writer.add_scalars(main_tag='TestAcc', tag_scalar_dict={file_name:acc}, global_step=epoch)
    return test_loss/args.batch_size , acc

if __name__ == '__main__':
    print('\n[Phase 3] : Training model')
    print('| Network Type = ' + str(args.net_type))
    print('| Training Epochs = ' + str(args.num_epochs))
    print('| Training Classes = ' + str(args.classes))
    print('| Initial Learning Rate = ' + str(args.lr))
    print('| Optimizer = ' + args.optimizer)
    print('| Loss Function = ' + args.loss)
    print('| Batch Size = ' + str(args.batch_size))
    print('| Widen Factor = ' + str(args.widen_factor))
    print('| Depth = ' + str(args.depth))
    print('| Start Epoch = ' + str(args.start_epoch))
    
    mlp.to(device)
    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'mse':
        criterion = nn.MSELoss()

    attack = None
    if args.training_type == 'fgsm':
        attack = FGSMAttack(model=mlp, epsilon=0.3, loss_fn=criterion)
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(mlp.parameters(), args.lr, weight_decay=args.weight_decay, momentum=args.momentum)            
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40000, 60000], gamma=0.1)

    elapsed_time = 0.0
    csvfile = open(os.path.join(csv_save_dir, 'data.csv'), 'w')
    csv_writer = csv.writer(csvfile, delimiter=' ')
    for epoch in range(args.start_epoch, args.start_epoch+args.num_epochs):
        start_time = time.time()
        train_loss, train_acc = train(epoch, optimizer, scheduler)
        test_loss, test_acc = test(epoch)
        epoch_time = time.time() - start_time
        print('| Epoch time : %d:%02d:%02d'  %(cf.get_hms(epoch_time)))
        elapsed_time += epoch_time
        csv_writer.writerow(['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])
        csv_writer.writerow([str(epoch), str(train_loss), str(test_loss), str(train_acc), str(test_acc)])
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))
    
    

    



