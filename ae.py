# Alan Cao
# December 5, 2019

# MNIST autoencoder for dimensional reduction

import os
import numpy
import torch
import math
from torch import nn
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as pyplot

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

def main():

    batch_size = 200
    num_epochs = 10
    learning_rate = 1e-3
    hidden_neurons = 144

    test_path = './ae'
    root = './data'

    model = ae(28*28, hidden_neurons)

    if not os.path.exists(test_path):
        os.mkdir(test_path)

    if not os.path.exists(root):
        os.mkdir(root)

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    transform = transforms.Compose([transforms.ToTensor()]) #convert to tensor
    train_set = datasets.MNIST(root=root, train=True, transform=transform, download=True) 
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    num_points = math.ceil(60000/batch_size)*num_epochs #number of epochs X number of batches
    trainloss = numpy.zeros(num_points)
    scaled_epoch = numpy.arange(0, num_epochs, num_epochs/num_points)
    point_count = 0

    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            img = img.view(img.size(0), -1) #matrix rows are different inputs
            img = Variable(img)
            if torch.cuda.is_available():
                img = img.cuda()

            # Forward Path
            out, hid = model(img)
            mse = torch.mean(img.sub(out)**2) #mean squared error
            loss = mse

            # Backward Path
            optimizer.zero_grad() #need to reset optimizer values
            loss.backward()
            optimizer.step()

            trainloss[point_count] = loss.item() #store loss values for later
            point_count = point_count + 1

        if (epoch + 1) % 1 == 0:     
            print('[{}/{}]:  loss:{:.4f}'.format(epoch + 1, num_epochs, loss.item()))

    pyplot.figure(1) #plot and save training loss
    pyplot.plot(scaled_epoch,trainloss)
    pyplot.legend(['loss'], loc='upper right')
    pyplot.xlabel('Number of Epochs')
    pyplot.savefig('{}/train_loss.png'.format(test_path))
    #pyplot.show()

    pyplot.figure(2) #plot and save first input of final batch
    img = img.view(img.size(0), 1, 28, 28)
    pyplot.imshow(img[0].cpu().numpy().squeeze(), cmap='gray_r')
    pyplot.axis('off')
    #pyplot.savefig('{}/iput_e{}.png'.format(test_path, epoch + 1))

    pyplot.figure(3) #plot and save first output of final batch
    out = out.view(img.size(0), 1, 28, 28)
    pyplot.imshow(out[0].cpu().detach().numpy().squeeze(), cmap='gray_r')
    pyplot.axis('off')
    #pyplot.savefig('{}/oput_e{}.png'.format(test_path, epoch + 1))

    # saves model as .pth file
    checkpoint = {
        'model': model,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}

    torch.save(checkpoint, '{}/ae.pth.tar'.format(test_path))

if __name__ == "__main__":
    main()
