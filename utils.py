from networks.mlp import *
from networks.resnet import *
from networks.wide_resnet import *
from ptflops import get_model_complexity_info
import matplotlib.pyplot as plt
import os
import scipy
import torch

class GetDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.labels = label
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label
    
    def __len__(self):
        return len(self.data)

def img_visualize(dataset=None):
    for i in range(1):
        img_arr, _ = dataset[i]
        img_arr = img_arr.resize(28, 28)
        file_name = '/home/xiangyu/AT/'+'mnist_train_%d.jpg'
        print(dataset.train_labels[i])
        scipy.misc.toimage(img_arr, cmin=0.0, cmax=1.0).save(file_name)

def get_model_info(net, input_size):
    #The input size should be (H, W, C)
    ops, params = get_model_complexity_info(net, input_size, as_strings=True, print_per_layer_stat=True, verbose=True)
    return ops, params

def getNetwork(args, params):
    
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
    elif (args.net_type == 'mlp'):
        net = MLP(params['input_size'], params['train_size']*args.classes*args.widen_factor, args.depth, args.classes)
        file_name = args.training_type+'-MLP-'+str(args.depth)+'-'+str(args.widen_factor)+'-'+str(args.classes)+'-'+args.optimizer
        if args.bn: 
            file_name += '-BN'
        else:
            pass
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet / MLP')
        sys.exit(0)

    return net, file_name

if __name__ == '__main__':
    net = MLP(in_plane=3072, inter_plane=250000, depth=1, out_class=10)
    opts, params = get_model_info(net, (1, 3072))
    print(opts)
    print(params)