import os

for f in os.listdir('./checkpoint/cifar10/std'):
    print(f.split('-0.3')[0])