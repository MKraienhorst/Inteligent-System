# -*- coding: utf-8 -*-
"""
Pytorch MNIST example

Created on Wed Nov 28 13:39:44 2018

@author: Gessert
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from glob import glob
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

# CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define architecture
        self.features = nn.Sequential(nn.Conv2d(1, 16, kernel_size=5, stride=2, bias=False,padding=2),
                                       nn.BatchNorm2d(16),
                                       nn.ReLU(),
                                       nn.Conv2d(16, 32, kernel_size=5,stride=2, bias=False, padding=2),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU())
        # Classifier
        self.gap = nn.AvgPool2d(7,1)
        self.classifier = nn.Linear(32,10)

    def forward(self, x):
        x = self.features(x)
        x = torch.squeeze(self.gap(x))
        x = self.classifier(x)
        return x
    
# Custom data loader
class MNISTDataset(Dataset):
    def __init__(self, image_path, ):
        # Get images
        self.im_paths = []
        labels_list = []
        all_folders = glob(image_path+'/*')
        for i in range(len(all_folders)):
            im_paths_curr = sorted(glob(all_folders[i]+'/*'))
            # define the labels
            labels_list.append(np.ones(len(im_paths_curr))*i)
            self.im_paths.extend(im_paths_curr)
        # Labels to matrix
        self.labels = np.concatenate(labels_list, axis=0) 

        # Preprocessing & Data augmentation
        self.composed = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])
            
    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        # Get image
        x = np.expand_dims(np.asarray(Image.open(self.im_paths[idx])),2)
        # Get label
        y = self.labels[idx]
        # Preprocessing & data augmentation
        x = self.composed(x)
        y = np.int(y)
        return x,y
    
# Training function
def train(model, device, train_loader, optimizer, loss_function, epoch):
    # Set to training mode
    model.train()
    # Loop over all examples
    for batch_idx, (data, target) in enumerate(train_loader):
        # Push to GPU
        data, target = data.to(device), target.to(device)
        # Reset gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # Calculate outputs
            output = model(data)
            # Calculate loss
            loss = loss_function(output, target)
            # Backpropagate loss
            loss.backward()
            # Apply gradients
            optimizer.step()
        if batch_idx % 50 == 0:
            print("Train Epoch:",epoch,"Loss:",loss.item())

            
# Testing function
def test(model, device, test_loader, loss_function):
    # Set to evaluation mode
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            curr_loss = loss_function(output,target)
            if i==0:
                predictions = output
                targets = target.data.cpu().numpy()
                loss = np.array([curr_loss.data.cpu().numpy()])
            else:
                predictions = np.concatenate((predictions,output))
                targets = np.concatenate((targets,target.data.cpu().numpy()))
                loss = np.concatenate((loss,np.array([curr_loss.data.cpu().numpy()])))
    # One-hot to normal:
    predictions = np.argmax(predictions,1)
    # Caluclate metrics
    accuracy = np.mean(np.equal(predictions,targets))
    conf_mat = confusion_matrix(targets,predictions)
    sensitivity = conf_mat.diagonal()/conf_mat.sum(axis=1)
    # Print metrics
    print("Test Accuracy",accuracy,"Test Sensitivity",np.mean(sensitivity),"Test loss",np.mean(loss))

if __name__ == "__main__":
    # Get datasets
    train_dataset = MNISTDataset("C:/PyTorch/ISM/mnist_png/mnist_png/training")
    train_loader = DataLoader(train_dataset,batch_size=500,shuffle=True,pin_memory=True)
    test_dataset = MNISTDataset("C:/PyTorch/ISM/mnist_png/mnist_png/testing")
    test_loader = DataLoader(test_dataset,batch_size=500,shuffle=False,pin_memory=True)
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Loss
    loss_function = nn.CrossEntropyLoss()
    # Model
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    # Train
    for epoch in range(1, 50 + 1):
        train(model, device, train_loader, optimizer, loss_function, epoch)
        test(model, device, test_loader, loss_function)    
               
            