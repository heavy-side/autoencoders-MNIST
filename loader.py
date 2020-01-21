# Alan Cao
# December 7, 2019

# simple code to load an existing network and run examples from MNIST

import os
import numpy
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as pyplot

from ae import ae
from sae import sae
from dae import dae
from dae import add_noise


test_path = './loader'
model_path = 'ae/ae.pth.tar'
#model_path = 'sae/sae.pth.tar'
#model_path = 'dae/dae.pth.tar'

noise_factor = 0.5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

checkpoint = torch.load(model_path)
model = checkpoint['model']
model.load_state_dict(checkpoint['state_dict'])
for parameter in model.parameters():
    parameter.requires_grad = False
model.eval() #ready for model evaluation

root = './data'
if not os.path.exists(root):
	os.mkdir(root)

if not os.path.exists(test_path):
	os.mkdir(test_path)

transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.MNIST(root=root, train=True , transform=transform, download=True) 
test_set  = datasets.MNIST(root=root, train=False, transform=transform, download=True) 

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_set,  batch_size=1, shuffle=False)

dataiter = iter(test_loader)
image, label = dataiter.next()
noise = add_noise(image, noise_factor)
image = image.to(device)
noise = noise.to(device)

image = image.view(image.size(0), -1) #matrix rows are different inputs
noise = noise.view(image.size(0), -1)
output, hidden = model(image)
output, hidden = model(noise)
image = image.view(image.size(0), 1, 28, 28) #back to a tensor of images
noise = noise.view(image.size(0), 1, 28, 28) #back to a tensor of images
output = output.view(image.size(0), 1, 28, 28) #back to a tensor of images

pyplot.figure(1) #inputs without noise
pyplot.subplot(1, 2, 1)
pyplot.imshow(image.cpu().numpy().squeeze(), cmap='gray_r')
pyplot.axis('off')
pyplot.title("Ground Truth")
pyplot.subplot(1, 2, 2)
pyplot.imshow(output.cpu().detach().numpy().squeeze(), cmap='gray_r')
pyplot.axis('off')
pyplot.title("Reconstruction")
pyplot.savefig('{}/compare.png'.format(test_path))

# pyplot.figure(2) #inputs with noise
# pyplot.subplot(1, 3, 1)
# pyplot.imshow(image.cpu().numpy().squeeze(), cmap='gray_r')
# pyplot.axis('off')
# pyplot.title("Ground Truth")
# pyplot.subplot(1, 3, 2)
# pyplot.imshow(noise.cpu().numpy().squeeze(), cmap='gray_r')
# pyplot.axis('off')
# pyplot.title("Network Input")
# pyplot.subplot(1, 3, 3)
# pyplot.imshow(output.cpu().detach().numpy().squeeze(), cmap='gray_r')
# pyplot.axis('off')
# pyplot.title("Reconstruction")
# #pyplot.savefig('{}/compare_noise.png'.format(test_path))

pyplot.show()