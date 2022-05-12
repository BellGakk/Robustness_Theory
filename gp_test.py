from networks.mlp import * 
from dataset import * 
import config as cf
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='gaussian process test')
parser.add_argument('--width', default=10, type=int)
parser.add_argument('--dataset', default='mnist', type=str)
parser.add_argument('--depth', default=1, type=int)
parser.add_argument('--classes', default=1, type=int)
parser.add_argument('--input_size', default=1, type=int)
parser.add_argument('--gp_num', default=100, type=int)
args = parser.parse_args()

writer = SummaryWriter(log_dir='./GP-log/')


def gp_test(dataset, gp_num):
    if dataset == 'mnist':
        trainset, testset = mnist()
    elif dataset == 'cifar10':
        trainset, testset = cifar10()
    elif dataset == 'cifar100':
        trainset, testset = cifar100()

    mlp = MLP(args.input_size, args.width * args.input_size, args.depth, args.classes)
    mlp.apply(param_init)
    mlp.eval()
    inputs = [x for x in torch.arange(-5, 5, 0.01)]
    for x in inputs:
        x = torch.reshape(x, (1,1))
        output = mlp(x).detach().numpy()
        writer.add_scalars(main_tag='GP-'+str(args.width)+'-'+str(args.depth), tag_scalar_dict={str(gp_num) : output}, global_step=x)

if __name__ == '__main__':
    for num in range(args.gp_num):
        print(num)
        gp_test(args.dataset, num)

