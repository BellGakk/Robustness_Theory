import math

std_training_params = {

    'start_epoch' : 1,
    'num_epochs'  : 200,
    'train_batch_size'  : 128,
    'test_batch_size'   : 100,
    'lr'          : 0.1,
    'momentum'    : 0.9,
    'weight_decay': 5e-4,
    'optim_type'  : 'SGD'

}

adv_training_params = {

    'start_epoch' : 1,
    'num_epochs'  : 76,
    'train_batch_size'  : 64,
    'test_batch_size'   : 50,
    'lr'          : 0.1,
    'momentum'    : 0.9,
    'weight_decay': 2e-4,
    'optim_type'  : 'SGD'

}

adv_example_params = {

    'step_size' : 0.007,
    'num_steps' : 10,
    'epsilon'   : 0.031,
    'beta'      : 6

}

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'mnist': (0.1307)
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'mnist': (0.3081)
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def learning_rate(epoch, args):
    optim_factor = 0
    if args.training_type == 'std' and args.net_type != 'mlp':
        if epoch > 160:
            optim_factor = 3
        elif epoch > 120:
            optim_factor = 2
        elif epoch > 60:
            optim_factor = 1
        return args.lr*math.pow(0.2, optim_factor)
    
    elif args.training_type == 'std' and args.net_type == 'mlp':
        if epoch > 750:
            optim_factor = 2
        elif epoch > 500:
            optim_factor = 1
        return args.lr*math.pow(0.2, optim_factor)

    elif args.training_type == 'trades':
        if epoch > 99:
            optim_factor = 3
        elif epoch > 89:
            optim_factor = 2
        elif epoch > 74:
            optim_factor = 1 
        return args.lr*math.pow(0.1, optim_factor)
    

def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s