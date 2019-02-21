'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
sys.path.append('./pytorch-cifar')
sys.path.append('./pytorch-cifar/models')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from pgd import attack


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--p', default=2, type=float, help='p-wasserstein distance')
parser.add_argument('--alpha', default=0.1, type=float, help='PGD step size')
parser.add_argument('--norm', default='linfinity')
parser.add_argument('--ball', default='wasserstein')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--reg', default=3000, type=float,
                    help='entropy regularization')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

checkpoint_dir = 'checkpoints'
checkpoint_file = f'./{checkpoint_dir}/cifar_lr_{args.lr}_reg_{args.reg}_p_{args.p}_alpha_{args.alpha}_norm_{args.norm}_ball_{args.ball}_epoch_{{}}.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
mu = torch.Tensor((0.4914, 0.4822, 0.4465)).unsqueeze(-1).unsqueeze(-1).to(device)
std = torch.Tensor((0.2023, 0.1994, 0.2010)).unsqueeze(-1).unsqueeze(-1).to(device)
unnormalize = lambda x: x*std + mu
normalize = lambda x: (x-mu)/std

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('==> regularization {}, p {}'.format(args.reg, args.p))

# Model
print('==> Building model..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
def test_nominal(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
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

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'

    resume_file = '{}/{}'.format(checkpoint_dir, args.resume)
    assert os.path.isfile(resume_file)
    checkpoint = torch.load(resume_file)
    
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch'] + 1
    print('==> start epoch {}'.format(start_epoch))
    test_nominal(start_epoch)
    checkpoint_file = './{}/mnist_lr_{}_p_{}_reg_{}_epoch_{}_resume_{}.pth'.format(checkpoint_dir, args.lr, args.p, args.reg, '{}', args.resume)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    nominal_correct = 0
    total_epsilon = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs_pgd, _, epsilons = attack(torch.clamp(unnormalize(inputs),min=0),
                                         targets, net, p=args.p, normalize=normalize, 
                                         epsilon_factor=1.5, epsilon=0.01, maxiters=50,
                                         epsilon_iters=5, 
                                         regularization=args.reg, 
                                         alpha=args.alpha, 
                                         norm=args.norm, 
                                         ball=args.ball)

        optimizer.zero_grad()
        outputs = net(normalize(inputs_pgd.detach()))
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        with torch.no_grad(): 
            outputs_nominal = net(inputs)
            _, predicted_nominal = outputs_nominal.max(1)
            nominal_correct += predicted_nominal.eq(targets).sum().item()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_epsilon += epsilons.sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Adv Acc: %.3f%% (%d/%d) | Acc: %.3f%% (%d/%d) | Eps: %.3f%%'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, 
                100.*nominal_correct/total, nominal_correct, total,
                total_epsilon/total))

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    nominal_correct = 0
    total = 0
    total_epsilon = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)

        inputs_pgd, _, epsilons = attack(torch.clamp(unnormalize(inputs),min=0), 
                                         targets, net, p = args.p, normalize=normalize, 
                                         epsilon_factor=1.5, epsilon=0.01, maxiters=50,
                                         epsilon_iters=5, 
                                         regularization=args.reg, 
                                         alpha=args.alpha, 
                                         norm=args.norm, 
                                         ball=args.ball)
        with torch.no_grad():
            outputs = net(normalize(inputs_pgd))
            loss = criterion(outputs, targets)

            outputs_nominal = net(inputs)
            _, predicted_nominal = outputs_nominal.max(1)
            nominal_correct += predicted_nominal.eq(targets).sum().item()

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_epsilon += epsilons.sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Adv Acc: %.3f%% (%d/%d) | Acc: %.3f%% (%d/%d) | Eps: %.3f%%'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, 
                100.*nominal_correct/total, nominal_correct, total, total_epsilon/total))

    # Save checkpoint.
    acc = 100.*correct/total
    eps = total_epsilon/total

    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc, 
        'eps': eps,
        'epoch': epoch,
    }
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(state, checkpoint_file.format(epoch))

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
