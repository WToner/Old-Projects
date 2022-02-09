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
import matplotlib.image as mpimg
from mpl_toolkits import mplot3d
from dataset import SimpleEntangled, Entangled, Curve
import cv2
import scipy.special

def nearest(colour, colours):
    distribution = np.zeros(len(colours))
    for i in range(len(colours)):
        vector = colours[i] - colour
        distance = np.linalg.norm(vector)
        distribution[i] = distance
    index = np.argmin(distribution)
    maximum = np.max(distribution)
    distribution = maximum-distribution
    distribution /= 0.1
    distribution = scipy.special.softmax(distribution)
    index = np.random.choice(len(colours),1, p=distribution)
    return colours[index[0]]


"""True Image is a numpy array size d*d*3, colours is a list of colours - each formed of 3 8-bit numbers"""
def initialise(true_image, colours):
    out = np.ones(true_image.shape)
    num = 0
    for i in range(true_image.shape[0]):
        for j in range(true_image.shape[1]):
            #rand = np.random.randint(0,len(colours)-1)
            color = nearest(true_image[i][j], colours)
            out[i][j] = color
            num += 1
            if num % 200 == 0:
                initial_tensor = torch.from_numpy(out)
                outy = initial_tensor.permute(2, 0, 1)
                utils.save_image(outy, "./Images/James/img"+str(int(num/200))+".png", normalize=True)
    return out

def iteration(real_image, gen_image, colours):
    top_left_x = random.randint(0, real_image.shape[0]-5)
    top_left_y = random.randint(0, real_image.shape[1]-5)
    #bottom_right_x = random.randint(top_left_x, real_image.shape[0]-1)
    #bottom_right_y = random.randint(top_left_y, real_image.shape[1]-1)
    bottom_right_x = top_left_x+4
    bottom_right_y = top_left_y+4
    total_color_real = np.asarray([0,0,0])
    total_color_gen  = np.asarray([0,0,0])
    size = (bottom_right_x-top_left_x)*(bottom_right_y-top_left_y)
    for i in range(top_left_x, bottom_right_x):
        for j in range(top_left_y, bottom_right_y):
            for k in range(3):
                total_color_real[k] += real_image[i][j][k]
                total_color_gen[k] += gen_image[i][j][k]
    old_distance = total_color_gen - total_color_real
    old_distance = np.linalg.norm(old_distance)

    new_gen = gen_image.copy()
    for i in range(top_left_x, bottom_right_x):
        for j in range(top_left_y, bottom_right_y):
            rand = random.randint(0,100)
            if rand == 0:
                rand2 = random.randint(0,len(colours)-1)
                new_gen[i][j] = colours[rand2]

    for i in range(top_left_x, bottom_right_x):
        for j in range(top_left_y, bottom_right_y):
            total_color_real += real_image[i][j]

    total_color_new = [0,0,0]
    for i in range(top_left_x, bottom_right_x):
        for j in range(top_left_y, bottom_right_y):
            total_color_new += new_gen[i][j]

    new_distance = total_color_new - total_color_real
    new_distance = np.linalg.norm(new_distance)

    if new_distance < old_distance:
        return new_gen
    else:
        return gen_image


def main():
    colours = []
    #for i in range(20):
    #    color1 = random.randint(0,255)
    #    color2 = random.randint(0,255)
    #    color3 = random.randint(0,255)
    #    colour = [color1, color2, color3]
    #    colours.append(colour)

    #colour1 = [255, 255, 255]
    #colour2 = [221, 231, 70]
    #colour3 = [254, 163, 33]
    #colour4 = [178, 228, 55]
    #colour5 = [255, 137, 171]
    #colour60 = [221, 162, 110]
   # colour6 = [191,223,234]
   # colour7 = [228, 5, 33]
   # colour8 = [0, 142, 208]
    colour1 = [255,255,255]
    colour2 = [224,11,11]
    colour3 = [72,11,224]
    colour4 = [216, 222, 35]
    colour5 = [35, 222, 60]
    colours.append(colour1)
    colours.append(colour2)
    colours.append(colour3)
    colours.append(colour4)
    colours.append(colour5)
    #colours.append(colour6)
    #colours.append(colour7)
    #colours.append(colour8)
    img = cv2.imread('./Images/MosaicKasiaresized.JPG')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    initial = initialise(img, colours=colours)
    initial_tensor = torch.from_numpy(initial)
    out = initial_tensor.permute(2, 0, 1)
    utils.save_image(out, "./Images/img.png", normalize=True)

    for i in range(out.size()[1]):
        line = ""
        for j in range(out.size()[1]):
            if j == 27:
                if out[0][i][j] == 255:
                    line = line + " 0 "
                elif out[0][i][j] == 216:
                    line = line + " 1 "
                elif out[0][i][j] == 35:
                    line = line + " 2 "
                elif out[0][i][j] == 72:
                    line = line + " 3 "
                else:
                    line = line + " 4 "
        if i < 28:
            print("This line", line)

    outimg2 = Image.fromarray(img, 'RGB')
    outimg2.save("./Images/img_true.png")
    for i in range(0):
        initial = iteration(img, initial, colours)
        print("Iteration Complete: ", str(i))
        if i % 1000 == 0:
            initial_tensor = torch.from_numpy(initial)
            out = initial_tensor.permute(2, 0, 1)
            utils.save_image(out, "./Images/img"+str(i)+".png", normalize=True)


if __name__ == '__main__':
    main()
