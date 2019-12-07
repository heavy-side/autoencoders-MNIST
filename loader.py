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
#from dae import dae
#from cae import cae
#from vae import vae

model_path = 'ae/mnist_ae.pth.tar'
#model_path = 'sae/mnist_sae.pth.tar'

checkpoint = torch.load(model_path)
model = checkpoint['model']
model.load_state_dict(checkpoint['state_dict'])
for parameter in model.parameters():
    parameter.requires_grad = False
model.eval() #ready for model evaluation

root = './data'
if not os.path.exists(root):
	os.mkdir(root)

transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.MNIST(root=root, train=True , transform=transform, download=True) 
test_set  = datasets.MNIST(root=root, train=False, transform=transform, download=True) 

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_set,  batch_size=1, shuffle=False)

dataiter = iter(train_loader)
image, label = dataiter.next()
if torch.cuda.is_available():
	image = image.cuda()

image = image.view(image.size(0), -1) #matrix rows are different inputs
output, hidden = model(image)
image = image.view(image.size(0), 1, 28, 28) #back to a tensor of images
output = output.view(image.size(0), 1, 28, 28) #back to a tensor of images

pyplot.figure(1) #plot and save input
pyplot.imshow(image.cpu().numpy().squeeze(), cmap='gray_r')
pyplot.axis('off')
#pyplot.savefig('input.png')

pyplot.figure(2) #plot and save output
pyplot.imshow(output.cpu().detach().numpy().squeeze(), cmap='gray_r')
pyplot.axis('off')
#pyplot.savefig('output.png')

pyplot.show()