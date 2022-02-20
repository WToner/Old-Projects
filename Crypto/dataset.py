import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.utils.data as data
from itertools import islice
import torch
import random
import csv


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

class Crypto(data.Dataset):
	def __init__(self, context_length=10, train_bool=True):
		#self.transform = transforms.Compose([						 ])

		self.context = context_length

		self.entangles = []
		#open, high, low, close, volume, count = self.getData()
		data = self.getData()
		self.seq_length = context_length
		train, targets = self.formTrainSeq(data, self.seq_length)
		train, test, train_targets, test_targets = self.split(train, targets)
		self.train = torch.Tensor(train)
		self.test = torch.Tensor(test)
		self.train_targets = torch.tensor(train_targets)
		self.test_targets = torch.tensor(test_targets)
		self.num_seq = len(train)
		self.train_bool = train_bool

	def split(self, data, targets):
		#train = []
		#test = []
		#train_targets = []
		#test_targets = []

		random.seed(4)
		random.shuffle(data)
		random.seed(4)
		random.shuffle(targets)
		num = len(data)
		train = data[0:int(num/2)]
		test = data[int(num/2):]

		train_targets = targets[0:int(num/2)]
		test_targets = targets[int(num/2):]

		return train, test, train_targets, test_targets

	def getData(self):
		with open('/afs/inf.ed.ac.uk/user/s18/s1873696/GitHub_STuff/test-run/Crypto/Crypto/kraken_BTC_USD_output.csv') as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')
			line_count = 0
			total_count = 0
			opens = []
			close = []
			low = []
			high = []
			volume = []
			for row in csv_reader:
				if line_count < 1480323 and line_count >4:
					opens.append(float(row[1]))
					close.append(float(row[4]))
					low.append(float(row[3]))
					high.append(float(row[2]))
					volume.append(float(row[5]))
					total_count += 1
				line_count = line_count+1
			print(f'Processed {line_count} lines.')
		return opens, high, low, close, volume, total_count

	"""At the moment we are predicting highs from highs which is retarded"""
	def formTrainSeq(self, data, sequence_length=10):
		open, high, low, close, volume, count = data
		train = []
		targets = []
		#elements = int(count/(sequence_length+1))
		for i in range(count-sequence_length-1):
			#train.append(high[i*sequence_length+i:(i+1)*sequence_length +i])
			#targets.append(high[(i+1)*sequence_length +i])
			train.append(high[i:sequence_length +i])
			targets.append(high[sequence_length +i+1])

		return train, targets

	def __getitem__(self, index):
		if self.train_bool:
			return self.train[index]/10000, self.train_targets[index]/10000
		else:
			return self.test[index] / 10000, self.test_targets[index] / 10000

	def __len__(self):
		return len(self.train)

dataset = Crypto()