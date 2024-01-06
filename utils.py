import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

def get_dataset(dataset_name, transform, batchsize):
    if dataset_name == 'flowers102':

        data_transform = transforms.Compose(transform)

        train = torchvision.datasets.Flowers102(root="./datasets/", download=True,
                                        transform=data_transform)

        trainloader = torch.utils.data.DataLoader(train, batch_size=batchsize,
                                                shuffle=True, num_workers=2)

        test = torchvision.datasets.Flowers102(root="./datasets/", download=True,
                                                transform=data_transform, split='test')

        testloader = torch.utils.data.DataLoader(train, batch_size=batchsize,
                                                shuffle=True, num_workers=2)

        return trainloader, testloader
    else:
        return None
    

def imshow(loader):
    dataiter = iter(loader)
    images, labels = next(dataiter)
    img = torchvision.utils.make_grid(images)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train(net, epochs, trainloader, criterion, optimizer, device):
    print("Device:", device)
    start_time = time.time()
    net = net.to(device)

    print('######### Starting Training ########')
    for epoch in tqdm(range(epochs), desc='Epochs'):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader, desc='Batch Progress'), 0):
            inputs, labels = data
            optimizer.zero_grad()
            #------------------------------------------------------
            #------------------------------------------------------
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
        
            #------------------------------------------------------
            #------------------------------------------------------
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('######### Finished Training ########')
    end_time = time.time()
    print('Total Trainig Time[s]: ', end_time - start_time, "\nAverage Training Time per Epoch [it/s]: ", (end_time-start_time)/epochs, "\nDevice:", device)
    return net

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')