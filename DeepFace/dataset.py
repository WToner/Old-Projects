import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.utils.data as data
import torch
import random
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import scipy
from scipy import linalg
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple


class Torus(data.Dataset):
	def __init__(self, nm_points=50, r1=1, r2=0.5):
		self.nm_points = nm_points
		self.r1 = r1
		self.r2 = r2
		if nm_points == 0:
			raise RuntimeError('Number of points cannot be zero')
		self.generateTorus()

	def generateTorus(self):
		theta = np.random.uniform(0, 1, self.nm_points)
		phi = np.random.uniform(0, 1, self.nm_points)
		theta, phi = np.meshgrid(theta, phi)

		x = (self.r1 + self.r2*np.cos(2.*np.pi*theta)) * np.cos(2.*np.pi*phi) 
		y = (self.r1 + self.r2*np.cos(2.*np.pi*theta)) * np.sin(2.*np.pi*phi)
		z = self.r1 * np.sin(2.*np.pi*theta)

		self.torus = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
		self.torus = self.torus.astype(np.float32)
		self.l_torus = self.torus[(self.torus[:,0]<0.0)]
		self.r_torus = self.torus[(self.torus[:,0]>=0.0)]

	def __getitem__(self, index):
		return self.torus[index]

	def __len__(self):
		return len(self.torus)

	def draw(self):
		fig = plt.figure()
		ax1 = fig.add_subplot(131, projection='3d')
		ax1.scatter(self.torus[:,0], self.torus[:,1], self.torus[:,2], c='r',marker='o')

		ax2 = fig.add_subplot(132, projection='3d')
		ax2.scatter(self.l_torus[:,0], self.l_torus[:,1], self.l_torus[:,2], marker='o')

		ax3 = fig.add_subplot(133, projection='3d')
		ax3.scatter(self.r_torus[:,0], self.r_torus[:,1], self.r_torus[:,2], marker='o')
		plt.show()

class LTorus(Torus):
	def __init__(self, nm_points=50, r1=1, r2=0.5):
		super(LTorus, self).__init__(nm_points, r1, r2)
	def __getitem__(self, index):
		return self.l_torus[index]
	def __len__(self):
		return len(self.l_torus)

class RTorus(Torus):
	def __init__(self, nm_points=50, r1=1, r2=0.5):
		super(RTorus, self).__init__(nm_points, r1, r2)
	def __getitem__(self, index):
		return self.r_torus[index]
	def __len__(self):
		return len(self.r_torus)

class Sphere(data.Dataset):
	def __init__(self, nm_points=50, r=1):
		self.nm_points = nm_points
		self.r = r
		if nm_points == 0:
			raise RuntimeError('Number of points cannot be zero')
		self.generateSphere()

	def generateSphere(self):
		theta = np.random.uniform(0, 1, self.nm_points)
		phi = np.random.uniform(0, 1, self.nm_points)
		theta, phi = np.meshgrid(theta, phi)

		x = self.r*np.sin(2.*np.pi*phi) * np.cos(2.*np.pi*theta) 
		y = self.r*np.sin(2.*np.pi*phi) * np.sin(2.*np.pi*theta)
		z = self.r * np.cos(2.*np.pi*phi)

		self.sphere = np.vstack([x.flatten(), y.flatten(), z.flatten()]).T
		self.sphere = self.sphere.astype(np.float32)
		self.l_sphere = self.sphere[(self.sphere[:,0]<0.0)]
		self.r_sphere = self.sphere[(self.sphere[:,0]>=0.0)]

	def __getitem__(self, index):
		return self.sphere[index]

	def __len__(self):
		return len(self.sphere)

	def draw(self):
		fig = plt.figure()
		ax1 = fig.add_subplot(131, projection='3d')
		ax1.scatter(self.sphere[:,0], self.sphere[:,1], self.sphere[:,2], c='r',marker='o')

		ax2 = fig.add_subplot(132, projection='3d')
		ax2.scatter(self.l_sphere[:,0], self.l_sphere[:,1], self.l_sphere[:,2], marker='o')

		ax3 = fig.add_subplot(133, projection='3d')
		ax3.scatter(self.r_sphere[:,0], self.r_sphere[:,1], self.r_sphere[:,2], marker='o')
		plt.show()

class LSphere(Sphere):
	def __init__(self,  nm_points=50, r=1):
		super(LSphere, self).__init__(nm_points, r)		
	def __getitem__(self, index):
		return self.r_sphere[index]
	def __len__(self):
		return len(self.r_sphere)

class RSphere(Sphere):
	def __init__(self,  nm_points=50, r=1):
		super(RSphere, self).__init__(nm_points, r)
	def __getitem__(self, index):
		return self.l_sphere[index]
	def __len__(self):
		return len(self.l_sphere)

"""Asif rotating ball class for sequential experiments"""
class RotatingBall(data.Dataset):
	def __init__(self, nm_seq=100, rows=32, columns=32, timesteps=20,
						radius=4):
		self.rows = rows
		self.columns = columns
		self.radius = radius
		self.nm_seq = nm_seq
		self.timesteps = timesteps
		self.balls = []

		self.blob = -1.*torch.ones((2*self.radius, 2*self.radius))
		for x in range(2*self.radius):
			for y in range(2*self.radius):
				if ((x-self.radius)**2 + (y-self.radius)**2 ) <= self.radius**2:
					self.blob[x,y] = 1.0
		self.genBalls()


	def genBalls(self):
		theta = torch.linspace(0, 1, self.timesteps)
		center_x = torch.randint(0+2*self.radius, self.columns-2*self.radius-1, (int(self.nm_seq**0.5),))
		center_y = torch.randint(0+2*self.radius, self.rows-2*self.radius-1, (int(self.nm_seq**0.5),))

		self.balls = []
		for cx in center_x:
			for cy in center_y:
				x = self.radius*np.cos(2.*np.pi*theta) + cx
				y = self.radius*np.sin(2.*np.pi*theta) + cy
				seq = []
				for xi, yi in zip(x, y):
					image = -1.*torch.ones(self.rows, self.columns)
					image[int(xi)-self.radius:int(xi)+self.radius, int(yi)-self.radius:int(yi) + self.radius] = self.blob
					seq.append(image)
				seq = torch.stack(seq)
				self.balls.append(seq)
		#arr = torch.stack(self.balls)
		#self.save_seq_gif(arr, out_path='./anim.gif')

	def save_seq_gif(self, seq, out_path):
		ncols = int(np.sqrt(seq.shape[0]))
		nrows = 1 + seq.shape[0]//ncols
		fig = plt.figure()
		camera = Camera(fig)
		for k in range(seq.shape[1]):
			for i in range(seq.shape[0]):
				ax = fig.add_subplot(nrows, ncols, i+1)
				ax.imshow(seq[i,k], animated=True)
				ax.set_axis_off()
			camera.snap()
		anim = camera.animate()
		anim.save('{}'.format(out_path), writer='imagemagick', fps=10)
		plt.cla()
		plt.clf()
		plt.close()
	def __getitem__(self, index):
		return self.balls[index]

	def __len__(self):
		return len(self.balls)

class BallDataset(data.Dataset):
	def __init__(self, rows=32, columns=32, channels=3, nm_samples=20000, radius=4):
		self.rows = rows
		self.columns = columns
		self.channels = channels
		self.nm_samples = nm_samples
		self.radius = radius
		self.balls = []
		self.redballs = []
		self.blueballs = []
		self.redball = torch.ones((self.channels, 2*self.radius, 2*self.radius))
		for x in range(2*self.radius):
			for y in range(2*self.radius):
				if ((x-self.radius)**2 + (y-self.radius)**2 ) < self.radius**2:
					self.redball[1:3,x,y] = 0

		self.blueball = torch.ones((self.channels, 2*self.radius, 2*self.radius))
		for x in range(2*self.radius):
			for y in range(2*self.radius):
				if ((x-self.radius)**2 + (y-self.radius)**2 ) < self.radius**2:
					self.blueball[0:2,x,y] = 0

		self.genBalls()

	def genBalls(self):
		for sample in range(self.nm_samples):
			image = torch.ones((self.channels, self.rows, self.columns))
			cx, cy = np.random.randint(0+self.radius,self.rows-self.radius),np.random.randint(0+self.radius, self.columns-self.radius)
			idx = np.random.uniform(0,1) < 1
			if idx:
				image[:, cx-self.radius:cx+self.radius, cy-self.radius:cy + self.radius] = self.redball
				self.redballs.append(image)
			else:
				image[:, cx-self.radius:cx+self.radius, cy-self.radius:cy + self.radius] = self.blueball
				self.blueballs.append(image)
		
			self.balls.append(image)
			#plt.imshow(np.transpose(self.balls[0].numpy(), (1,2,0)))

	def __getitem__(self, index):
		return self.balls[index]

	def __len__(self):
		return len(self.balls)

class RedBalls(BallDataset):
	def __init__(self, rows=32, columns=32, channels=3, nm_samples=20000, radius=4):
		super(RedBalls, self).__init__(rows, columns, channels, nm_samples, radius)
	def __getitem__(self, index):
		return self.redballs[index]
	def __len__(self):
		return len(self.redballs)

class BlueBalls(BallDataset):
	def __init__(self, rows=32, columns=32, channels=3, nm_samples=20000, radius=4):
		super(BlueBalls, self).__init__(rows, columns, channels, nm_samples, radius)
	def __getitem__(self, index):
		return self.blueballs[index]
	def __len__(self):
		return len(self.blueballs)

class Entangled(data.Dataset):
	def __init__(self, nm_samples=20000):
		self.nm_samples = nm_samples

		self.entangles = []

		self.genBalls()
		#self.visualise()

	def genBalls(self):
		for sample in range(self.nm_samples):
			idx = np.random.uniform(0, 1) < 0.5
			up = np.random.uniform(0, 1) < -0.1
			veryup = np.random.uniform(0, 1) < -0.1
			coords = torch.zeros(2)
			if idx:
				x = np.random.uniform(-3,0)
				y = np.random.uniform(-3,3)
				if y*y + x*x < 9 and y*y + x*x > 4:
					coords[0] = x
					if up:
						y = y+5
					if veryup:
						y = y+10
					coords[1] = y
					self.entangles.append((coords,0))
				else:
					sample = sample-1
			else:
				x = np.random.uniform(0,3)
				y = np.random.uniform(-3,3)
				if y*y + x*x < 9 and y*y + x*x > 4:
					coords[0] = x-1
					if up:
						y = y+5
					if veryup:
						y = y+10
					coords[1] = y+2.5
					self.entangles.append((coords,1))
				else:
					sample = sample-1

	def visualise(self):
		visual = []
		sample = 0
		while sample < self.nm_samples:
			idx = np.random.uniform(0, 1) < 0.5
			up = np.random.uniform(0, 1) < -0.1
			veryup = np.random.uniform(0, 1) < -0.1
			coords = np.array([0.1,0.1])
			if idx:
				x = np.random.uniform(-3,0)
				y = np.random.uniform(-3,3)
				if y*y + x*x < 9 and y*y + x*x > 4:
					coords[0] = x
					if up:
						y = y+5
					if veryup:
						y = y+10
					coords[1] = y
					visual.append(coords)
					sample += 1
			else:
				x = np.random.uniform(0,3)
				y = np.random.uniform(-3,3)
				if y*y + x*x < 9 and y*y + x*x > 4:
					coords[0] = x-1
					if up:
						y = y+5
					if veryup:
						y = y+10
					coords[1] = y+2.5
					visual.append(coords)
					sample += 1
		visual_np = np.array(visual)
		# print(visual_np[0:10,0])
		plt.scatter(visual_np[:,0], visual_np[:,1])
		plt.show()
		return visual_np


	def __getitem__(self, index):
		return self.entangles[index]

	def __len__(self):
		return len(self.entangles)

class SimpleEntangled(data.Dataset):
	def __init__(self, nm_samples=20000):
		self.nm_samples = nm_samples

		self.entangles = []

		self.genSamples()
		#self.visualise()

	def genSamples(self):
		while len(self.entangles) < self.nm_samples:
			coords = torch.zeros(2)
			x = np.random.uniform(-3,0)
			y = np.random.uniform(-3,3)
			if y*y + x*x < 9 and y*y + x*x > 4:
				coords[0] = x
				coords[1] = y
				self.entangles.append((coords,0))

	def visualise(self):
		visual = []
		sample = 0
		while sample < self.nm_samples:
			coords = np.array([0.1,0.1])
			x = np.random.uniform(-3,0)
			y = np.random.uniform(-3,3)
			if y*y + x*x < 9 and y*y + x*x > 4:
				coords[0] = x
				coords[1] = y
				visual.append(coords)
				sample += 1
		visual_np = np.array(visual)
		# print(visual_np[0:10,0])
		plt.scatter(visual_np[:,0], visual_np[:,1])
		plt.show()
		return visual_np


	def __getitem__(self, index):
		return self.entangles[index]

	def __len__(self):
		return len(self.entangles)

class SimpleDisjoint(data.Dataset):
	def __init__(self, nm_samples=20000):
		self.nm_samples = nm_samples

		self.samples = []
		self.genSamples()
		#self.visualise()

	def genSamples(self):
		while len(self.samples) < self.nm_samples:
			coords = torch.zeros(2)
			x = np.random.uniform(-2,2)
			y = np.random.uniform(-2,2)
			if y*y + x*x < 4:
				rand = random.randint(0,3)
				if rand == 0:
					coords[0] = x + 3
					coords[1] = y + 3
				elif rand == 1:
					coords[0] = x + 3
					coords[1] = y - 3
				elif rand == 2:
					coords[0] = x - 3
					coords[1] = y + 3
				else:
					coords[0] = x - 3
					coords[1] = y - 3
				self.samples.append((coords,0))

	def visualise(self):
		visual = []
		sample = 0
		while sample < self.nm_samples:
			coords = np.array([0.1,0.1])
			x = np.random.uniform(-3,0)
			y = np.random.uniform(-3,3)
			if y*y + x*x < 9 and y*y + x*x > 4:
				coords[0] = x
				coords[1] = y
				visual.append(coords)
				sample += 1
		visual_np = np.array(visual)
		# print(visual_np[0:10,0])
		plt.scatter(visual_np[:,0], visual_np[:,1])
		plt.show()
		return visual_np


	def __getitem__(self, index):
		return self.samples[index]

	def __len__(self):
		return len(self.samples)

class Ball(data.Dataset):
	def __init__(self, centre_x=0, centre_y = 0, radius=1, nm_samples=20000):
		self.nm_samples = nm_samples
		self.centre_x = centre_x
		self.centre_y = centre_y
		self.radius = radius

		self.entangles = []
		self.genSamples()

	def genSamples(self):
		while len(self.entangles) < self.nm_samples:
			coords = torch.zeros(2)
			x = np.random.uniform(-self.radius,self.radius)
			y = np.random.uniform(-self.radius,self.radius)
			if y*y + x*x < self.radius*self.radius:
				coords[0] = x+self.centre_x
				coords[1] = y+self.centre_y
				self.entangles.append((coords,0))

	def __getitem__(self, index):
		return self.entangles[index]

	def __len__(self):
		return len(self.entangles)

class Curve(data.Dataset):
	def __init__(self, centre_x=0, centre_y = 0, radius=2, nm_samples=20000):
		self.nm_samples = nm_samples
		self.centre_x = centre_x
		self.centre_y = centre_y
		self.radius = radius

		self.entangles = []
		self.genSamples()

	def genSamples(self):
		while len(self.entangles) < self.nm_samples:
			theta = np.random.uniform(0,np.pi)
			coords = torch.zeros(2)
			x = self.radius*np.cos(theta)
			y = self.radius*np.sin(theta)
			coords[0] = x+self.centre_x
			coords[1] = y+self.centre_y
			self.entangles.append((coords,0))

	def __getitem__(self, index):
		return self.entangles[index]

	def __len__(self):
		return len(self.entangles)

"""We start off by making this the upper sphere"""
class Manifold(data.Dataset):
	def __init__(self, centre_x=0, centre_y = 0, centre_z = 0, radius=4, nm_samples=10000):
		self.nm_samples = nm_samples
		self.centre_x = centre_x
		self.centre_y = centre_y
		self.centre_z = centre_z
		self.radius = radius

		self.entangles = []
		self.genSamples()

	def genSamples(self):
		while len(self.entangles) < self.nm_samples:
			coords = torch.zeros(3)
			x = np.random.uniform(-self.radius,self.radius)
			y = np.random.uniform(-self.radius,self.radius)
			if y*y + x*x < self.radius*self.radius:
				coords[0] = x+self.centre_x
				coords[1] = y+self.centre_y
				z = self.radius*self.radius - x*x - y*y
				coords[2] = np.sqrt(z)+self.centre_z
				self.entangles.append((coords,0))

	def __getitem__(self, index):
		return self.entangles[index]

	def __len__(self):
		return len(self.entangles)

"""The issue is that at the moment this does not constitute a uniform distribution on the surface."""
class Sphere(data.Dataset):
	def __init__(self, centre_x=0, centre_y = 0, centre_z = 0, radius=4, nm_samples=10000):
		self.nm_samples = nm_samples
		self.centre_x = centre_x
		self.centre_y = centre_y
		self.centre_z = centre_z
		self.radius = radius

		self.entangles = []
		self.genSamples()

	def genSamples(self):
		while len(self.entangles) < self.nm_samples:
			coords = torch.zeros(3)
			x = np.random.uniform(-self.radius,self.radius)
			y = np.random.uniform(-self.radius,self.radius)
			if y*y + x*x < self.radius*self.radius:
				coords[0] = x+self.centre_x
				coords[1] = y+self.centre_y
				up = np.random.uniform(0, 1) < 0.5   #we need to change this back to 0.5
				z = self.radius * self.radius - x * x - y * y
				if up:
					coords[2] = np.sqrt(z) + self.centre_z
				else:
					coords[2] = -np.sqrt(z) + self.centre_z

				self.entangles.append((coords,0))

	def __getitem__(self, index):
		return self.entangles[index]

	def __len__(self):
		return len(self.entangles)

class UniformSphere(data.Dataset):
	def __init__(self, centre_x=0, centre_y = 0, centre_z = 0, radius=4, nm_samples=10000):
		self.nm_samples = nm_samples
		self.centre_x = centre_x
		self.centre_y = centre_y
		self.centre_z = centre_z
		self.radius = radius

		self.entangles = []
		self.genSamples()

	def genSamples(self):
		while len(self.entangles) < self.nm_samples:
			coords = torch.zeros(3)
			x = np.random.uniform(-1,1)
			y = np.random.uniform(-1,1)
			z = np.random.uniform(0,1)   #This means that we're dealing with a half sphere atm
			radius_squared = x*x + y*y + z*z
			if radius_squared < 1:
				radius = np.sqrt(radius_squared)
				x = x/radius
				y = y/radius
				z = z/radius
				coords[0] = x*self.radius+self.centre_x
				coords[1] = y * self.radius + self.centre_y
				coords[2] = z * self.radius + self.centre_z
				self.entangles.append((coords,0))

	def __getitem__(self, index):
		return self.entangles[index]

	def __len__(self):
		return len(self.entangles)

class OneDimensional(data.Dataset):
	def __init__(self, nm_samples=20000):
		self.nm_samples = nm_samples

		self.samples = []
		self.genSamples()
		#self.visualise()

	def genSamples(self):
		while len(self.samples) < self.nm_samples:
			randint = random.randint(0,2)

			if randint == 0:
				x = np.random.normal(-4,2)
			else:
				x = np.random.normal(3, 2)
			self.samples.append((x,0))

	def visualise(self):
		visualise = []
		while len(visualise) < self.nm_samples:
			randint = random.randint(0,1)
			if randint == 0:
				x = np.random.normal(-2,1)
			else:
				x = np.random.normal(2, 1)
			visualise.append(x)
		visual_np = np.array(visualise)
		print(visual_np.shape)
		plt.hist(visual_np, bins =30)
		plt.show()
		return visual_np


	def __getitem__(self, index):
		return self.samples[index]

	def __len__(self):
		return len(self.samples)

def pil_loader(path: str) -> Image.Image:
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')

transform = transforms.Compose([
	#transforms.Resize(240,240),
	transforms.CenterCrop((240,240)),
	transforms.Resize((94,94)),
    transforms.ToTensor()
])

class Landmark(data.Dataset):
	def __init__(self, filename="/disk/scratch/william/Face/landmarks_trump.txt", train=True):
		self.filename = filename

		#filename="/disk/scratch/william/Face/landmarks_trump.txt", train=True):
		#filename="/home/william/Documents/Work/landmarks_trump.txt", train=True):

		self.train = train

		self.images = []
		self.labels = []
		if train:
			self.genSamples(0,8000)
			#self.genSamples(0,200)
		else:
			self.genSamples(8000, 10000)

		test_row = self.labels[52]
		self.ref_row = test_row.reshape(-1, 2)
		#plt.scatter(test_row[:,0], 1-test_row[:,1])
		#plt.savefig("./trumpfig"+str(i)+".png")
		#plt.clf()
		#plt.show()

	def get_ref(self):
		return self.ref_row

	def get_mean_bot_left(self):
		mean = [0,0]
		for i in range(len(self.labels)):
			mean[0] = mean[0] + self.labels[i][0]
			mean[1] = mean[1] + self.labels[i][1]
		mean[0] = mean[0]/len(self.labels)
		mean[1] = mean[1]/len(self.labels)
		return mean

	def get_mean_box_size(self):
		sum = 0
		for i in range(len(self.labels)):
			sum = sum + self.labels[i][2]-self.labels[i][0]
		sum = sum/len(self.labels)
		return sum

	def genSamples(self, a, b):
		with open(self.filename) as f:
			lines = f.readlines()
			#for i in range(len(lines)):
			for i in range(a, b):
				array = lines[i].split()
				if not array[0][21:23] == "17":
					path = "/disk/scratch/william/Face/" + array[0]
					#path = "/home/william/Documents/Work/" + array[0]
					img = pil_loader(path)
					self.images.append(transform(img))
					if "/" in str(array[1:]):
						if "/" in str(array[2:]):
							np_array = np.asarray(array[3:])
							self.labels.append(np_array.astype(np.float))
						else:
							np_array = np.asarray(array[2:])
							self.labels.append(np_array.astype(np.float))
					else:
						np_array = np.asarray(array[1:])
						self.labels.append(np_array.astype(np.float))

	def __getitem__(self, index):
		return self.images[index], self.labels[index]

	def __len__(self):
		return len(self.images)

#dataset = Landmark()


class Target_Landmark(data.Dataset):
	def __init__(self, filename="/disk/scratch/william/Face/landmarks_target.txt"):
		self.filename = filename

		#filename="/home/william/Documents/Work/pytorch_face_landmark/landmarks_target.txt"
		#filename="/disk/scratch/william/Face/landmarks_target.txt"
		self.indices = []
		self.labels = []
		self.genSamples()

		self.labels = np.asarray(self.labels)
		self.indices = np.asarray(self.indices)
		self.indices = np.expand_dims(self.indices,1)
		self.labels = np.concatenate((self.indices, self.labels), 1)

		#test_row = self.labels[12,1:]
		test_row = self.labels[50, 1:]
		self.ref_row = test_row.reshape(-1, 2)
		#plt.scatter(test_row[:,0], 1-test_row[:,1])
		#plt.savefig("./fig"+str(i)+".png")
		#plt.clf()
		#plt.show()

		self.labels = self.labels[np.argsort(self.labels[:,0])]

	def get_ref(self):
		return self.ref_row

	def get_mean_bot_left(self):
		mean = [0,0]
		for i in range(len(self.labels)):
			mean[0] = mean[0] + self.labels[i][1]
			mean[1] = mean[1] + self.labels[i][2]
		mean[0] = mean[0]/len(self.labels)
		mean[1] = mean[1]/len(self.labels)
		return mean

	def get_mean_box_size(self):
		sum = 0
		for i in range(len(self.labels)):
			sum = sum + self.labels[i][3]-self.labels[i][1]
			#print(self.labels[i][1], self.labels[i][3], self.labels[i][3]-self.labels[i][1])
		sum = sum/len(self.labels)
		return sum

	def genSamples(self,):
		with open(self.filename) as f:
			lines = f.readlines()
			for i in range(len(lines)):
				array = lines[i].split()
				number = int(array[0][21:25])
				self.indices.append(number)
				if "/" in str(array[1:]):
					if "/" in str(array[2:]):
						np_array = np.asarray(array[3:])
						self.labels.append(np_array.astype(np.float))
					else:
						np_array = np.asarray(array[2:])
						self.labels.append(np_array.astype(np.float))
				else:
					np_array = np.asarray(array[1:])
					self.labels.append(np_array.astype(np.float))

	def __getitem__(self, index):
		return self.labels[index][1:]

	def __len__(self):
		return len(self.labels)

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

	#shifted = np.concatenate((shifted_rim, shifted_brow, shifted_nose, shifted_eyes_l, shifted_eyes_r, shifted_mouth), 0)
	return a_rim, a_brows, a_nose, a_eyes_l, a_eyes_r, a_mouth
	#return shifted

def normalise(target_ref, normaliser):
	a_rim, a_brows, a_nose, a_eyes_l, a_eyes_r, a_mouth = normaliser
	target_rim = target_ref[0:17]
	target_nose = target_ref[27:36]
	target_eyes_l = target_ref[36:42]
	target_eyes_r = target_ref[42:48]
	target_brows = target_ref[17:27]
	target_mouth = target_ref[48:]

	shifted_rim = np.matmul(target_rim, a_rim)
	shifted_nose = np.matmul(target_nose, a_nose)
	shifted_eyes_l = np.matmul(target_eyes_l, a_eyes_l)
	shifted_eyes_r = np.matmul(target_eyes_r, a_eyes_r)
	shifted_brow = np.matmul(target_brows, a_brows)
	shifted_mouth = np.matmul(target_mouth, a_mouth)

	shifted = np.concatenate((shifted_rim, shifted_brow, shifted_nose, shifted_eyes_l, shifted_eyes_r, shifted_mouth),
							 0)

	return shifted

"""
target_dataset = Target_Landmark()
dataset = Landmark()

trump_edge = dataset.get_mean_box_size()
trump_box_left = dataset.get_mean_bot_left()

target_edge = target_dataset.get_mean_box_size()
target_box_left = target_dataset.get_mean_bot_left()

print(trump_box_left)
shift = [0,0]
shift[0] = trump_box_left[0] - target_box_left[0]
shift[1] = trump_box_left[1] - target_box_left[1]
scale = trump_edge/target_edge

print(scale, shift)"""

"""
trump_ref = dataset.get_ref()

target_ref = target_dataset.get_ref()
ones = np.ones((target_ref.shape[0], 1))
target_ref = np.concatenate((target_ref, ones), 1)

normaliser = prepare_normaliser(trump_ref, target_ref)
shifted = normalise(target_ref, normaliser)
plt.scatter(shifted[:,0], 1-shifted[:,1])
plt.show()

for i in range(100):
	target_test = target_dataset.__getitem__(i)
	target_test = np.concatenate((target_test.reshape(-1, 2), ones), 1)
	shifted = normalise(target_test, normaliser)
	plt.scatter(shifted[:,0], 1-shifted[:,1])
	plt.savefig("./stretched_fig"+str(i)+".png")
	plt.clf()
	#plt.show()
"""
