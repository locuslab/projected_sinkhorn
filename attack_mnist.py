'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import sys
sys.path.append('./pytorch-cifar')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

# from models import *
from utils import progress_bar
from pgd import attack


parser = argparse.ArgumentParser(description='PyTorch MNIST attack')
parser.add_argument('--reg', default=1000, type=float,
                    help='entropy regularization')
parser.add_argument('--p', default=2, type=float, help='p-wasserstein distance')
parser.add_argument('--alpha', default=0.1, type=float, help='PGD step size')
parser.add_argument('--norm', default='linfinity')
parser.add_argument('--ball', default='wasserstein')
parser.add_argument('--checkpoint')
parser.add_argument('--binarize', action='store_true')
args = parser.parse_args()

if args.checkpoint is None: 
    raise ValueError('Need checkpoint file to attack')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

net = nn.Sequential(
        nn.Conv2d(1, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(32*7*7,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

net = net.to(device)

regularization = args.reg
print('==> regularization set to {}'.format(regularization))

model_name = './checkpoints/{}'.format(args.checkpoint)
save_name = './epsilons/{}_reg_{}_p_{}_alpha_{}_norm_{}_ball_{}.pth'.format(
                args.checkpoint, regularization, args.p, 
                args.alpha, args.norm, args.ball)
binarize = args.binarize

print('==> loading model {}'.format(model_name))
print('==> saving epsilon to {}'.format(save_name))
d = torch.load(model_name)
if 'state_dict' in d: 
    net.load_state_dict(d['state_dict'][0])

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
elif 'robust' in model_name: 
    net.load_state_dict(d)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
else: 

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(d['net'])

criterion = nn.CrossEntropyLoss()

def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            if binarize: 
                inputs = (inputs >= 0.5).float()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total

def test_attack(): 
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_epsilons = []

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        if binarize: 
            inputs = (inputs >= 0.5).float()

        inputs_pgd, _, epsilons = attack(torch.clamp(inputs,min=0), targets, net,  
                                         regularization=regularization,
                                         p=args.p, 
                                         alpha=args.alpha, 
                                         norm=args.norm, 
                                         ball=args.ball,
                                         epsilon=0.7, maxiters=200, kernel_size=7)
        
        outputs_pgd = net(inputs_pgd)
        loss = criterion(outputs_pgd, targets)

        test_loss += loss.item()
        _, predicted = outputs_pgd.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        epsilons[predicted == targets] = -1
        all_epsilons.append(epsilons)

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Avg epsilon: %.3f'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, torch.cat(all_epsilons).float().mean().item()))

        acc = 100.*correct/total
        torch.save((acc, torch.cat(all_epsilons)), save_name)

test()
test_attack()
