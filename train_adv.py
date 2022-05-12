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
from at_type.trade import *

parser = argparse.ArgumentParser(description='Adversarial Training')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=34, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', default=False, help='resume from checkpoint')
parser.add_argument('--training_type', default='trades', type=str, help='standarding training or adversarial training.')
parser.add_argument('--log_intervals', default=10, help='batch intervals for logging the training result ')

args = parser.parse_args()
adv_training_params = cf.adv_training_params
adv_example_params = cf.adv_example_params
if torch.cuda.is_available():
    device = torch.device('cuda:0')
best_acc = 0.0

print('[Phase 0] : Hyperparameters confirmed')
print('Adversarial Training Params:')
for key in adv_training_params.keys():
    sys.stdout.write('| '+key+' : '+str(adv_training_params[key])+'\n')
print('Adversarial Example Params:')
for key in adv_example_params.keys():
    sys.stdout.write('| '+key+' : '+str(adv_example_params[key])+'\n')

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

trainloader = torch.utils.data.DataLoader(trainset, batch_size=adv_training_params['train_batch_size'], shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=adv_training_params['test_batch_size'], shuffle=False, num_workers=2)

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


if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    net, file_name = getNetwork(args)
    resume_checkpoint_dir = './checkpoint'+os.sep+args.dataset+os.sep+args.training_type+os.sep+file_name
    assert os.path.isdir(resume_checkpoint_dir), 'Error: No checkpoint directory found!'
    epoch_exist = 0
    for f in os.listdir(resume_checkpoint_dir):
        if f.split('.')[1] == 'pt':
            new_epoch_check = int((f.split('epoch')[1]).split('.')[0])
            if new_epoch_check >= epoch_exist:
                epoch_exist = new_epoch_check
    net_state_dict = torch.load(resume_checkpoint_dir+os.sep+file_name+'-epoch'+str(epoch_exist)+'.pt')['net']
    net.load_state_dict(net_state_dict)

else:
    start_epoch = 1
    net, file_name = getNetwork(args)
    net.apply(conv_init)
    print('| Building net type [' + file_name+']...')

# Model
print('\n[Phase 2] : Model setup')
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

def train_trades(epoch):
    global best_acc
    net.train()
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(0.1, epoch, 'trades'), momentum=0.9, weight_decay=5e-4)
    print('optimizer initialized for Epoch{} in Trades.'.format(epoch))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        # calculate robust loss
        loss = trades_loss(model=net,
                           x_natural=inputs,
                           y=targets,
                           optimizer=optimizer,
                           step_size=adv_example_params['step_size'],
                           epsilon=adv_example_params['epsilon'],
                           perturb_steps=adv_example_params['num_steps'],
                           beta=adv_example_params['beta'])
        loss.backward()
        optimizer.step()
        # print progress
        if batch_idx % args.log_intervals == 0:
            sys.stdout.write('\r')
            sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f'%(
                epoch, adv_training_params['num_epochs'], batch_idx+1, (len(trainset)//adv_training_params['train_batch_size'])+1,
                       loss.item()))
            sys.stdout.flush()

    print('================================================================')
    train_loss, train_accuracy = trades_eval_train()
    test_loss, test_accuracy = trades_eval_test()
    print('================================================================')

    if test_accuracy > best_acc:
        state_dict = {
            'net' : net.state_dict(),
            'acc' : test_accuracy,
            'epoch' : epoch,
            }
        torch.save(state_dict, os.path.join(model_type_point, file_name+'-epoch{}.pt'.format(epoch)))
        best_acc = test_accuracy


def trades_eval_train():
    net.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            train_loss += F.cross_entropy(outputs, targets, size_average=False).item()
            preds = outputs.max(1, keepdim=True)[1]
            correct += preds.eq(targets.view_as(preds)).sum().item()
    train_loss /= len(trainloader.dataset)
    sys.stdout.write('\r')
    sys.stdout.write('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss, correct, len(trainloader.dataset),
        100. * correct / len(trainloader.dataset)))
    sys.stdout.flush()
    train_accuracy = correct / len(trainloader.dataset)
    return train_loss, train_accuracy

def trades_eval_test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            test_loss += F.cross_entropy(outputs, targets, size_average=False).item()
            preds = outputs.max(1, keepdim=True)[1]
            correct += preds.eq(targets.view_as(preds)).sum().item()
    test_loss /= len(testloader.dataset)
    sys.stdout.write('\r')
    sys.stdout.write('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    sys.stdout.flush()
    test_accuracy = correct / len(testloader.dataset)
    return test_loss, test_accuracy

if __name__ == '__main__':

    print('\n[Phase 3] : Training model')
    print('| Training Epochs = ' + str(adv_training_params['num_epochs']))
    print('| Initial Learning Rate = ' + str(adv_training_params['lr']))
    print('| Optimizer = ' + adv_training_params['optim_type'])

    criterion = nn.CrossEntropyLoss()
    #net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])

    net = net.to(device)
    
    elapsed_time = 0.0
    for epoch in range(start_epoch, start_epoch+adv_training_params['num_epochs']):
        start_time = time.time()
        if args.training_type == 'trades':
            train_trades(epoch)
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))



