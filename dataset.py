import torch
import torchvision
import os
from torchvision import datasets, transforms
import numpy as np
import cv2
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)
def mnist():
    mnist_train = datasets.MNIST(root='./mnist_data/', 
                                 transform=transform_train,
                                 train=True,
                                 download=True)
    mnist_test = datasets.MNIST(root='./mnist_data/',
                                transform=transform_test,
                                train=False,
                                download=True)
    return mnist_train, mnist_test

def cifar10():
    cifar10_train = datasets.CIFAR10(root='./cifar10_data/',
                                   transform=transform_train,
                                   train=True,
                                   download=True)
    cifar10_test = datasets.CIFAR10(root='./cifar10_data/',
                                  transform=transform_test,
                                  train=False,
                                  download=True)
    return cifar10_train, cifar10_test

def cifar100():
    cifar100_train = datasets.CIFAR100(root='./cifar100_data/',
                                       transform=transform_train,
                                       train=True,
                                       download=True)
    cifar100_test = datasets.CIFAR100(root='./cifar100_data/',
                                       transform=transform_test,
                                       train=False,
                                       download=True)
    return cifar100_train, cifar100_test

def main():
    mnist_train, mnist_test = mnist()
    cifar10_train, cifar10_test = cifar10()
    cifar100_train, cifar100_test = cifar100()

    mnist_train_loader = torch.utils.data.DataLoader(dataset=mnist_train, 
                                                    batch_size=1,
                                                    shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                                    batch_size=1,
                                                    shuffle=True)
    cifar10_train_loader = torch.utils.data.DataLoader(dataset=cifar10_train,
                                                       batch_size=1,
                                                       shuffle=True)
    cifar10_test_loader = torch.utils.data.DataLoader(dataset=cifar10_test,
                                                      batch_size=1,
                                                      shuffle=True)
    cifar100_train_loader = torch.utils.data.DataLoader(dataset=cifar100_train,
                                                        batch_size=1,
                                                        shuffle=True)
    cifar100_test_loader = torch.utils.data.DataLoader(dataset=cifar100_test,
                                                       batch_size=1,
                                                       shuffle=True)
    print('The length of MNIST\'s training dataset is '+str(len(mnist_train_loader)))
    print('The length of MNIST\'s testing dataset is '+str(len(mnist_test_loader)))
    print('The length of CIFAR10\'s training dataset is '+str(len(cifar10_train_loader)))
    print('The length of CIFAR10\'s testing dataset is '+str(len(cifar10_test_loader)))
    print('The length of CIFAR100\'s training dataset is '+str(len(cifar100_train_loader)))
    print('The length of CIFAR100\'s testing dataset is '+str(len(cifar100_test_loader)))
    for idx, (inputs, targets) in enumerate(mnist_test_loader):
        print(inputs.data)
    return mnist_train_loader, mnist_test_loader, cifar10_train_loader, cifar10_test_loader, cifar100_train_loader, cifar100_test_loader
    
if __name__ == '__main__':
    main()
