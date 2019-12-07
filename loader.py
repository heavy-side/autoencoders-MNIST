# Alan Cao
# October 9, 2019

# MNIST dataset loading example
# MNIST dataset contains 60000 training images and 10000 testing images
# look towards EMNIST for larger dataset

import os
import numpy
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as pyplot

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

root = './data'
if not os.path.exists(root):
	os.mkdir(root)

transform=transforms.Compose([ transforms.ToTensor(), transforms.Normalize( (1.0,) , (1.0,))])

# if not exist, download datasets
train_set = datasets.MNIST(root=root, train=True , transform=transform, download=True) 
test_set  = datasets.MNIST(root=root, train=False, transform=transform, download=True) 

batch_size = 100

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_set,  batch_size=batch_size, shuffle=False)

print('==>>> total training images:', '60000')
print('==>>> total testing images :', '10000')

print('==>>> total training batches:', (len(train_loader)))
print('==>>> total testing batches :', (len(test_loader )))

dataiter = iter(train_loader)
num_image, num_label = dataiter.next()

print('==>>> MNIST image batch shape:', num_image.shape)
print('==>>> MNIST label batch shape:', num_label.shape)

if 0: # example image from MNIST 
	pyplot.imshow(num_image[0].numpy().squeeze(), cmap='gray_r')
	pyplot.axis('off')
	pyplot.show()
	
if 0: # example image grid from MNIST
	figure = pyplot.figure()
	grid_images = 5 # choose in interval [1,10]
	for index in range(1, grid_images**2 + 1):
		pyplot.subplot(5, 5, index)
		pyplot.axis('off')
		pyplot.imshow(num_image[index].numpy().squeeze(), cmap='gray_r')
	pyplot.show()