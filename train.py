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

from networks.wide_resnet import *
from networks.resnet import *
from torch.autograd import Variable
from dataset import *

parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=101, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=16, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', default=False, help='resume from checkpoint')
parser.add_argument('--training_type', default='trades', type=str, help='standard supervised training or adversarial training.')
parser.add_argument('--log_intervals', default=20, help='intervals for logging the training result')

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0.0
args = parser.parse_args()
start_epoch, num_epochs, batch_size, lr, momentum, weight_decay, optim_type = cf.training_config(args.training_type)
if args.training_type != 'std': 
    step_size, num_steps, epsilon, beta = cf.adv_config(args.training_type)


# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = args.training_type+'-lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = args.training_type+'-vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = args.training_type+'-resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = args.training_type+'-wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    net.apply(conv_init)

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

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

# Training
def train_std(epoch):
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(lr, epoch), momentum=momentum, weight_decay=weight_decay)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        
        if batch_idx % args.log_intervals == 0:
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' %(epoch, num_epochs, batch_idx+1,
                    (len(trainset)//batch_size)+1, loss.item(), 100.*correct/total))
            sys.stdout.flush()

def test_std(epoch):
    global best_acc
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))

        if acc > best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
            state = {
                    'net':net.module if use_cuda else net,
                    'acc':acc,
                    'epoch':epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'+args.dataset+os.sep
            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            torch.save(state, save_point+file_name+'.t7')
            best_acc = acc

def train_trades(epoch):
    global best_acc
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(lr, epoch, args.training_type), momentum=momentum, weight_decay=weight_decay)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            device = torch.device('cuda:0')
            inputs, targets = inputs.cuda(), targets.cuda()
        else:
            device = torch.device('cpu')

        optimizer.zero_grad()
        # calculate robust loss
        loss = trades_loss(model=net,
                           x_natural=inputs,
                           y=targets,
                           optimizer=optimizer,
                           step_size=step_size,
                           epsilon=epsilon,
                           perturb_steps=num_steps,
                           beta=beta)
        loss.backward()
        optimizer.step()
        # print progress
        if batch_idx % args.log_interval == 0:
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f'%(
                epoch, num_epochs, batch_idx+1, (len(trainset)//batch_size)+1,
                       loss.item()))
            sys.stdout.write()

    print('================================================================')
    train_loss, train_accuracy = trades_eval_train(net, device, trainloader)
    test_loss, test_accuracy = trades_eval_test(net, device, testloader)
    print('================================================================')

    if test_accuracy > best_acc:
        torch.save(net.state_dict(), os.path.join(model_type_point, file_name+'-epoch{}.pt'.format(epoch)))
        best_acc = test_accuracy


def trades_eval_train():
    net.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in trainloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            train_loss += F.cross_entropy(outputs, targets, size_average=False).item()
            preds = outputs.max(1, keepdim=True)[1]
            correct += preds.eq(targets.view_as(preds)).sum().item()
    train_loss /= len(trainloader.dataset)
    sys.stdout.write('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(trainloader.dataset),
        100. * correct / len(trainloader.dataset)))
    training_accuracy = correct / len(trainloader.dataset)
    return train_loss, train_accuracy

def trades_eval_test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets, size_average=False).item()
            preds = outputs.max(1, keepdim=True)[1]
            correct += preds.eq(targets.view_as(preds)).sum().item()
    test_loss /= len(testloader.dataset)
    sys.stdout.write('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    test_accuracy = correct / len(testloader.dataset)
    return test_loss, test_accuracy

def main():
    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(lr))
    print('| Optimizer = ' + str(optim_type))
    elapsed_time = 0
    for epoch in range(start_epoch, start_epoch+num_epochs):
        start_time = time.time()
        if args.training_type == 'std':
            train_std(epoch)
            test_std(epoch)
        elif args.training_type == 'trades':
            train_trades(epoch)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))
