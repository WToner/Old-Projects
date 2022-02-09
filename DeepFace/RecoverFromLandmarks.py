from __future__ import print_function
import argparse
import torch
import math
import numpy as np
from torch.nn import init
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
from mpl_toolkits import mplot3d
from dataset import Landmark, Target_Landmark
import scipy
from scipy import linalg

def prepare_normaliser(trump_ref, target_ref):
    trump_rim = trump_ref[0:17]
    trump_nose = trump_ref[27:36]
    trump_eyes_l = trump_ref[36:42]
    trump_eyes_r = trump_ref[42:48]
    trump_brows = trump_ref[17:27]
    trump_mouth = trump_ref[48:]

    target_rim = target_ref[0:17]
    target_nose = target_ref[27:36]
    target_eyes_l = target_ref[36:42]
    target_eyes_r = target_ref[42:48]
    target_brows = target_ref[17:27]
    target_mouth = target_ref[48:]

    a_rim, _, _, _ = scipy.linalg.lstsq(target_rim, trump_rim)
    a_nose, _, _, _ = scipy.linalg.lstsq(target_nose, trump_nose)
    a_eyes_l, _, _, _ = scipy.linalg.lstsq(target_eyes_l, trump_eyes_l)
    a_eyes_r, _, _, _ = scipy.linalg.lstsq(target_eyes_r, trump_eyes_r)
    a_brows, _, _, _ = scipy.linalg.lstsq(target_brows, trump_brows)
    a_mouth, _, _, _ = scipy.linalg.lstsq(target_mouth, trump_mouth)

    #shifted_rim = np.matmul(target_rim, a_rim)
    #shifted_nose = np.matmul(target_nose, a_nose)
    #shifted_eyes_l = np.matmul(target_eyes_l, a_eyes_l)
    #shifted_eyes_r = np.matmul(target_eyes_r, a_eyes_r)
    #shifted_brow = np.matmul(target_brows, a_brows)
    #shifted_mouth = np.matmul(target_mouth, a_mouth)

    #shifted = np.concatenate((shifted_rim, shifted_brow, shifted_nose, shifted_eyes_l, shifted_eyes_r, shifted_mouth),
    #                         0)
    return a_rim, a_nose, a_eyes_l, a_eyes_r, a_brows, a_mouth

def normalise(x, normaliser, device):
    a_rim, a_nose, a_eyes_l, a_eyes_r, a_brows, a_mouth = normaliser

    batch_size = x.size()[0]
    x = x[:,4:]
    y = x[:,:4]
    x = x.view(batch_size, 68, 2)
    ones = torch.ones(batch_size, 68, 1).cuda(7)
    x = torch.cat((x,ones), dim=2)

    x_rim = x[:,0:17,:]
    x_nose = x[:,27:36,:]
    x_eyes_l = x[:,36:42,:]
    x_eyes_r = x[:,42:48,:]
    x_brow = x[:,17:27,:]
    x_mouth = x[:,48:,:]

    x_rim = x_rim.reshape(batch_size*17,3)
    x_nose = x_nose.reshape(batch_size*9,3)
    x_eyes_l = x_eyes_l.reshape(batch_size*6,3)
    x_eyes_r = x_eyes_r.reshape(batch_size*6,3)
    x_brow = x_brow.reshape(batch_size*10,3)
    x_mouth = x_mouth.reshape(batch_size*20,3)

    shifted_rim = np.matmul(x_rim.cpu(), a_rim)
    shifted_nose = np.matmul(x_nose.cpu(), a_nose)
    shifted_eyes_l = np.matmul(x_eyes_l.cpu(), a_eyes_l)
    shifted_eyes_r = np.matmul(x_eyes_r.cpu(), a_eyes_r)
    shifted_brow = np.matmul(x_brow.cpu(), a_brows)
    shifted_mouth = np.matmul(x_mouth.cpu(), a_mouth)

    shifted_rim = shifted_rim.view(batch_size, 17, 2)
    shifted_nose = shifted_nose.view(batch_size, 9, 2)
    shifted_eyes_l = shifted_eyes_l.view(batch_size, 6, 2)
    shifted_eyes_r = shifted_eyes_r.view(batch_size, 6, 2)
    shifted_brow = shifted_brow.view(batch_size, 10, 2)
    shifted_mouth = shifted_mouth.view(batch_size, 20, 2)

    shifted = np.concatenate((shifted_rim, shifted_brow, shifted_nose, shifted_eyes_l, shifted_eyes_r, shifted_mouth),1)
    shifted = shifted.reshape(batch_size, 68*2)
    shifted = torch.from_numpy(shifted)
    shifted = shifted.to(device)
    shifted = torch.cat((y, shifted), dim=1)
    return shifted

class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        #self.fc1 = nn.Linear(136, )
        self.conv1 = nn.Conv2d(3,400, kernel_size=4, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(400)
        self.conv2 = nn.Conv2d(400,200, kernel_size=4, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(200)
        self.conv3 = nn.Conv2d(200,100, kernel_size=4, stride=2, padding=0)
        self.bn3 = nn.BatchNorm2d(100)
        self.conv4 = nn.Conv2d(100,50, kernel_size=4, stride=2, padding=0)
        self.fc1 = nn.Linear(50*4*4, 1)


    def forward(self, x):
        batch_size = x.size()[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.5)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.5)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.5)

        x = self.conv4(x)

        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = F.sigmoid(x)
        out = x.sum(dim=0)
        out = out/batch_size

        return out

#354 or 240
class Decoder(nn.Module):
    def __init__(self, ):
        super(Decoder, self).__init__()
        #self.fc1 = nn.Linear(136, )
        self.conv1 = nn.ConvTranspose2d(140,800, kernel_size=4, stride=2, padding=0, output_padding=0)
        self.bn1 = nn.BatchNorm2d(800)
        self.conv2 = nn.ConvTranspose2d(800,400, kernel_size=4, stride=2, padding=0, output_padding=0)
        self.bn2 = nn.BatchNorm2d(400)
        self.conv3 = nn.ConvTranspose2d(400,200, kernel_size=4, stride=2, padding=0, output_padding=0)
        self.bn3 = nn.BatchNorm2d(200)
        self.conv4 = nn.ConvTranspose2d(200,100, kernel_size=4, stride=2, padding=0, output_padding=0)
        self.bn4 = nn.BatchNorm2d(100)
        self.conv5 = nn.ConvTranspose2d(100,3, kernel_size=4, stride=2, padding=0, output_padding=0)


    def forward(self, x):
        batch_size = x.size()[0]
        #batch_size = x.shape[0]

        x = x.view(batch_size, 140,1,1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.5)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.5)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.5)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x, 0.5)

        x = self.conv5(x)
        x = F.tanh(x)
        #print(x.size())

        return x

def train(args, decoder, discriminator, device, train_loader, optimizer, optimizer_disc, epoch=0):
    decoder.train()
    discriminator.train()
    total_loss = 0
    i = 0
    for image, landmark in train_loader:
        rand = random.randint(0,1)
        #landmark = torch.from_numpy(landmark)
        landmark = landmark.to(device, dtype=torch.float)
        if rand == 0:
            batch_size = landmark.shape[0]
            optimizer.zero_grad()
            decoder.zero_grad()
            image = image.to(device)

            out = decoder(landmark)
            loss_recon = F.mse_loss(out, image, reduction='sum')
            fake_stat = discriminator(out)
            real_stat = discriminator(image)
            loss_disc = F.mse_loss(fake_stat, real_stat, reduction='sum')
            loss = loss_recon + loss_disc
            loss.backward()
            optimizer.step()
            total_loss += loss
            i += 1
        else:
            batch_size = landmark.shape[0]
            optimizer_disc.zero_grad()
            discriminator.zero_grad()
            image = image.to(device)

            out = decoder(landmark)
            fake_stat = discriminator(out.detach())
            real_stat = discriminator(image)
            loss_disc = -F.mse_loss(fake_stat, real_stat, reduction='sum')

            loss_disc.backward()
            optimizer_disc.step()

    total_loss = total_loss/(i*batch_size)
    print("Training Loss Epoch " + str(epoch) + " is: " + str(total_loss.item()))

def test(args, decoder, device, test_loader, epoch=0):
    decoder.eval()
    total_loss = 0
    i = 0
    for image, landmark in test_loader:
        batch_size = landmark.shape[0]
        decoder.zero_grad()
        image = image.to(device)
        landmark = landmark.to(device)

        out = decoder(landmark.float())
        loss = F.mse_loss(out, image, reduction='sum')
        total_loss += loss
        if i == 0:
            print('saving the output')
            utils.save_image(out.detach(), './Images/fake_samples_epoch_' + str(int(epoch)) + '.png',
                                 normalize=True)
        i += 1
    total_loss = total_loss/(i*batch_size)
    print("Test Loss Epoch " + str(epoch) + " is: " + str(total_loss.item()))

def target(args, normaliser, shift, scale, decoder, device, target_loader, epoch=0):
    decoder.eval()
    i = 0
    for landmark in target_loader:
        edges = landmark[:, 2] - landmark[:, 0]
        edges = edges*scale
        landmark[:,0] = landmark[:,0] + shift[0]
        landmark[:,1] = landmark[:,1] + shift[1]
        landmark[:,2] = landmark[:,0] + edges
        landmark[:, 3] = landmark[:, 1] + edges
        landmark = landmark.to(device)
        landmark = normalise(landmark, normaliser, device)
        #landmark = torch.from_numpy(landmark)
        landmark = landmark.to(device, dtype=torch.float)
        out = decoder(landmark)
        if i == 0:
            print('saving the Target output')
            for i in range(50):
                utils.save_image(out[i].detach(), './Images/me_samples_iter_' + str(i) + '.png', normalize=True)
        i += 1


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
    help='input batch size for training (default: 64)')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--cuda_no', type=int, default=7, help='using gpu device id')
    parser.add_argument('--id', type=int, default=1, help='identifier to add to filenames')
    parser.add_argument('--L_type', type=int, default=2, help='Type of L-loss L1 or L2')
    parser.add_argument('--eta', type=float, default=1, metavar='M',
                        help='How much to weight disc loss')
    parser.add_argument('--weight-decay', type=float, default=0.02, metavar='N',
    help='input weight decay (default: 0.01)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
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
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
    help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.cuda_no) if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset = Landmark(train=True)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=30, num_workers=1, shuffle=True)

    dataset_test = Landmark(train=False)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=30, num_workers=1, shuffle=True)

    decoder = Decoder().to(device)
    optimizer = torch.optim.RMSprop(decoder.parameters(), lr=args.lr)

    discriminator = Discriminator().to(device)
    optimizer_disc = torch.optim.RMSprop(discriminator.parameters(), lr=0.1*args.lr)

    target_dataset = Target_Landmark()
    target_loader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size=50, num_workers=1, shuffle=True)

    #decoder.load_state_dict(torch.load("/disk/scratch/william/Face/params/decoder_170"))

    trump_ref = dataset.get_ref()

    target_ref = target_dataset.get_ref()
    ones = np.ones((target_ref.shape[0], 1))
    target_ref = np.concatenate((target_ref, ones), 1)

    trump_edge = dataset.get_mean_box_size()
    trump_box_left = dataset.get_mean_bot_left()

    target_edge = target_dataset.get_mean_box_size()
    target_box_left = target_dataset.get_mean_bot_left()

    shift = [0,0]
    shift = np.asarray(shift)
    shift[0] = trump_box_left[0] - target_box_left[0]
    shift[1] = trump_box_left[1] - target_box_left[1]
    scale = trump_edge / target_edge

    normaliser = prepare_normaliser(trump_ref[4:], target_ref[4:])

    for epoch in range(1, args.epochs):
        train(args, decoder, discriminator, device, train_loader, optimizer, optimizer_disc, epoch=epoch)
        if epoch % 5 == 0:
            target(args, normaliser, shift, scale, decoder, device, target_loader, int(epoch/10))
            test(args, decoder, device, test_loader, epoch=int(epoch/10))
            #torch.save(decoder.state_dict(), "/disk/scratch/william/Face/params/decoder_" + str(epoch))



if __name__ == '__main__':
    main()
