# Alan Cao
# October 27, 2019

# MNIST binarized dataset autoencoder for dimensional reduction

import os
import numpy
import torch
import math
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as pyplot

root = './data'
if not os.path.exists(root):
	os.mkdir(root)

batch_size = 1
num_epochs = 10
learning_rate = 1e-3

def normalize(tensor):             #lambda function for image binarization
    tensor = tensor - tensor.min() #set minimum to zero
    tensor = tensor / tensor.max() #set NEW maximum to one
    tensor = torch.round(tensor)   #round to nearest integer value
    return tensor

transform = transforms.Compose([
    transforms.ToTensor(),                              #convert to tensor
    #transforms.Lambda(lambda tensor:normalize(tensor)), #normalize and round
])

train_set = datasets.MNIST(root=root, train=True, transform=transform, download=True) 
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

class ae(nn.Module):
    def __init__(self, inputs, hidden):
        super(ae, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(inputs, hidden),
            nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Linear(hidden, inputs),
            nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

num_points = math.ceil(60000/batch_size)*num_epochs #number of epochs X number of batches
trainloss1 = numpy.zeros(num_points)
trainloss2 = numpy.zeros(num_points)
scaled_epoch = numpy.arange(0, num_epochs, num_epochs/num_points)
point_count = 0
fig_count = 0

test_path = './test'
if not os.path.exists(test_path):
	os.mkdir(test_path)

model = ae(28*28, 128)
criterion1 = nn.BCELoss() #binary cross entropy
criterion2 = nn.MSELoss() #mean squared error
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1) #matrix rows are different inputs
        img = Variable(img)

        # Forward Path
        out, hid = model(img)
        loss1 = criterion1(out, img) #BCE
        loss2 = criterion2(out, img) #MSE

        # Backward Path
        optimizer.zero_grad() #need to reset optimizer values
        #loss1.backward()
        loss2.backward()
        optimizer.step()

        trainloss1[point_count] = loss1.item() #store loss values for later
        trainloss2[point_count] = loss2.item()

#        if (point_count) % 6000 == 0:
#            pyplot.figure(fig_count)
#            img = img.view(img.size(0), 1, 28, 28)
#            pyplot.imshow(img[0].numpy().squeeze(), cmap='gray_r')
#            pyplot.axis('off')
#            pyplot.savefig('{}/iput_p{}.png'.format(test_path, point_count + 1))

#            pyplot.figure(fig_count + 1)
#            out = out.view(img.size(0), 1, 28, 28)
#            pyplot.imshow(out[0].detach().numpy().squeeze(), cmap='gray_r')
#            pyplot.axis('off')
#            pyplot.savefig('{}/oput_p{}.png'.format(test_path, point_count + 1))
#            fig_count = fig_count + 2
#            pyplot.close("all")

        point_count = point_count + 1
        
    print('[{}/{}]:  bce:{:.4f};  mse:{:.4f}'.format(epoch + 1, num_epochs, loss1.item(), loss2.item()))

    if (epoch + 1) % 1 == 0: #prints and example input and output every kth epoch
        pyplot.figure(fig_count)
        pyplot.figure(1)
        img = img.view(img.size(0), 1, 28, 28)
        pyplot.imshow(img[0].numpy().squeeze(), cmap='gray_r')
        pyplot.axis('off')
        pyplot.savefig('{}/iput_e{}.png'.format(test_path, epoch + 1))

        pyplot.figure(fig_count + 1)
        out = out.view(img.size(0), 1, 28, 28)
        pyplot.imshow(out[0].detach().numpy().squeeze(), cmap='gray_r')
        pyplot.axis('off')
        pyplot.savefig('{}/oput_e{}.png'.format(test_path, epoch + 1))
        fig_count = fig_count + 2
        pyplot.close("all")

pyplot.figure(fig_count)
pyplot.plot(scaled_epoch,trainloss1)
pyplot.plot(scaled_epoch,trainloss2)
pyplot.legend(['BCE', 'MSE'], loc='upper right')
pyplot.xlabel('Number of Epochs')
pyplot.savefig('{}/train_loss.png'.format(test_path))
pyplot.show()

#torch.save(model.state_dict(), './mnist_ae_test3.pth')   # saves model as .pth file

