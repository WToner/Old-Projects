from __future__ import print_function
import argparse
import torch
import math
import numpy as np
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import random
from torchvision import datasets, transforms
from numpy.random import seed
from numpy.random import randint
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from dataset import Crypto


def tensor_to_image(x, name):
    image = x[:, :, :, :].detach().cpu().numpy()
    image = image * 0.5 + 0.5
    plt.imshow(image[0][0])
    # img = Image.fromarray(np.uint8(image[0][0] * 28) , 'L')
    plt.savefig(f'{name}_img.png')
    return


class Classifier(nn.Module):
    def __init__(self, context=10):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(context, 1)

    def forward(self, x):
        batch_size = x.size()[0]

        #print(self.fc1.weight[0])
        x = self.fc1(x)

        return x


def train(args, classifier, device, train_loader, optimizer, epoch):
    classifier.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.unsqueeze(dim=1)
        batch_size = data.size()[0]
        context = data.size()[1]
        optimizer.zero_grad()

        output = classifier(data)

        classifier_loss = F.mse_loss(output, target)
        classifier_loss.backward()
        optimizer.step()

        #if batch_idx % args.log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #               100. * batch_idx / len(train_loader), classifier_loss.item()))



def test(args, classifier, device, test_loader):
    classifier.eval()
    profit = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.unsqueeze(dim=1)
            context = data.size()[1]

            output = classifier(data)

            last = data[:, context - 1]
            last = last.unsqueeze(dim=1)
            diff = target - last
            signs = torch.sign(output - last)

            classifier_loss = signs * diff
            profit += classifier_loss.sum(dim=0)
            #print(classifier_loss.sum(dim=0).item())

        #test_loss /= len(test_loader.dataset)
    profit = profit*100
    print('\nTest set: Average Profit: {:.4f}\n'.format(profit.item()))


def test_on_train(args, classifier, device, test_loader, text_file):
    classifier.eval()
    profit = 0
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.unsqueeze(dim=1)
            context = data.size()[1]

            output = classifier(data)

            last = data[:, context - 1]
            last = last.unsqueeze(dim=1)
            diff = target - last
            signs = torch.sign(output - last)

            classifier_loss = signs * diff
            mse_loss = F.mse_loss(output, target)
            profit += classifier_loss.sum(dim=0)
            total_loss += mse_loss
            #print(classifier_loss.sum(dim=0).item())

    text_file.write(str(total_loss.item())+ "\n")

        #test_loss /= len(test_loader.dataset)
    profit = profit*100
    print('\nTrain set: Average Profit: {:.4f}, MSE Loss {}\n'.format(profit.item(), total_loss.item()))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=640, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--weight-decay', type=float, default=0.02, metavar='N',
                        help='input weight decay (default: 0.01)')
    parser.add_argument('--test-batch-size', type=int, default=640, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=100, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    context_length = 100

    dataset = Crypto(context_length=context_length, train_bool=True)
    dataset_test = Crypto(context_length=context_length, train_bool=False)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, num_workers=1, shuffle=True)

    classifier = Classifier(context=context_length).to(device)

    optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    output_file = open("/home/infuser/Documents/code/test-run/Crypto/train_loss.txt", "w")

    for epoch in range(1, args.epochs + 1):
        train(args, classifier, device, train_loader, optimizer, epoch)
        #test(args, classifier, device, test_loader)
        test_on_train(args, classifier, device, train_loader, output_file)

    if (args.save_model):
        torch.save(model.state_dict(), "baseline1_model.pt")

    output_file.close()


if __name__ == '__main__':
    main()