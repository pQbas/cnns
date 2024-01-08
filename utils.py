import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import os


def get_dataset(dataset_name, transform, batchsize):
    
    if dataset_name == 'flowers102':

        data_transform = transforms.Compose(transform)

        train = torchvision.datasets.Flowers102(root="./datasets/", download=True,
                                        transform=data_transform)

        trainloader = torch.utils.data.DataLoader(train, batch_size=batchsize,
                                                shuffle=True, num_workers=2)

        test = torchvision.datasets.Flowers102(root="./datasets/", download=True,
                                                transform=data_transform, split='test')

        testloader = torch.utils.data.DataLoader(test, batch_size=batchsize,
                                                shuffle=True, num_workers=2)

        return trainloader, testloader
    
    if dataset_name == 'cifar10':

        data_transform = transforms.Compose(transform)

        train = torchvision.datasets.CIFAR10(root="./datasets/", 
                                             train=True,
                                             transform=data_transform,
                                             download=True)

        trainloader = torch.utils.data.DataLoader(train, batch_size=batchsize,
                                                shuffle=True, num_workers=2)
        
        test = torchvision.datasets.CIFAR10(root="./datasets/", 
                                             train=False,
                                             transform=data_transform,
                                             download=True)
        
        testloader = torch.utils.data.DataLoader(test, batch_size=batchsize,
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

def train(net, epochs, trainloader, criterion, optimizer, device, every_n_epochs=10):
    print("Device:", device)
    start_time = time.time()
    net = net.to(device)
    accuracy_hist = []
    
    print('######### Starting Training ########')
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        train_acc = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            #------------------------------------------------------
            #------------------------------------------------------
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            max_scores, max_idx = outputs.max(dim=1)
            train_acc += torch.sum(max_idx == labels)/len(labels)


            #------------------------------------------------------
            #------------------------------------------------------
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            torch.cuda.empty_cache()

            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            #     running_loss = 0.0

        train_acc = train_acc.item()
        accuracy_hist.append(train_acc/(i+1))
        
        if epoch % every_n_epochs == 0:
            print(f"it:{epoch}/{epochs}, Average Accuracy:{train_acc/(i+1):.3f}")


    print('######### Finished Training ########')
    end_time = time.time()
    print('Total Trainig Time[s]: ', end_time - start_time, "\nAverage Training Time per Epoch [it/s]: ", (end_time-start_time)/epochs, "\nDevice:", device)


    plt.plot(accuracy_hist)
    plt.ylabel('Average Accuracy')
    plt.show()

    return net

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    


def create_directory(directory_path):
    ####################################################################################
    # Replace 'your_model_directory' with the desired path for your model directory
    # create_directory(model_directory)
    ####################################################################################

    try:
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{directory_path}' already exists.")



def save_model(path, model, weights='model.pth'):
    torch.save(model.state_dict(), os.path.join(path,weights))
