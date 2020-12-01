# -*- coding: utf-8 -*-
"""
Code is depending on: Nils Gessert's Pytorch MNIST example
other sources are: 
Stefan Otte https://github.com/sotte/pytorch_tutorial/blob/master/notebooks/03_transfer_learning.ipynb
pytorch example: https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
Nils Gessert's: https://github.com/ngessert/isic2018 and the paper for general ideas 

Created on Wed Nov 28 13:39:44 2018

@author: Gessert
"""
import torchvision.models as models
# resnet18 = models.resnet18()
# alexnet = models.alexnet()
# vgg16 = models.vgg16()
# squeezenet = models.squeezenet1_0()
# densenet = models.densenet121()
# inception = models.inception_v3()

import shutil
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
#from glob import glob
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import configparser
import os
import csv

# parameters
#put everything here into init file
_lr = 0.001  # define learning rate
_numModel = 1 # _numModel == 1 -> densenet; others are not working
_bin = 0 #if _bin == 1 -> use binary classification 
_onlyEval = 0
_numEpos = 5 #number of training/evaluation cycles
_batchSize = 3 #number of batch size -> how many images are loaded together to training/validation
_pathTrain = (r"..\ISIC2018_Task3_Training_Input_test")

# #do not use the init file right now
# open the configuration file and initialise variables
#config = configparser.ConfigParser()
#config.read('./Documents/init.ini')
#FeatureMatrixName = config['DEFAULT']['FeatureMatrixName']
##segementationInt = int(config['Bool']['useSegementation'])
#if(segementationInt == 1): #use segementation 1 = yes, 0 = no
#    useSegementation = True
#else:
#    useSegementation = False 
#removeHairInt = int(config['Bool']['useRemoveHair'])
#if(removeHairInt == 1):
#    useRemoveHair = True
#else:
#    useRemoveHair = False


# get groundtruth data from csv file and put it into a matrix
# first line is still the definition of the label (so skip one line)
def getGroundTruth(path):

    with open(path) as csvfile:  ####### adjust the path!!#######
        readCSV = csv.reader(csvfile, delimiter=',')
        labels1 = []
        labels2 = []
        labels3 = []
        labels4 = []
        labels5 = []
        labels6 = []
        labels7 = []
        
        for row in readCSV:
            label1 = [row[1]]
            labels1.append(label1) 
            label2 = [row[2]]
            labels2.append(label2)
            label3 = [row[3]]
            labels3.append(label3)
            label4 = [row[4]]
            labels4.append(label4)
            label5 = [row[5]]
            labels5.append(label5)
            label6 = [row[6]]
            labels6.append(label6)
            label7 = [row[7]]
            labels7.append(label7)
        
        labels = [labels1,labels2,labels3,labels4,labels5,labels6,labels7]
        labels = list(map(list, zip(*labels)))
        labels = np.asarray(labels)
        labels = labels[1:len(labels),:]
        labelsArray = []
        if(_bin == 0): # bin == 0 do not use binary classification
            print("7 classes prediction")
            classAmounts = np.zeros([7])
            weights = np.zeros([7])
            for i in range(len(labels)):
                temp = np.argmax(labels[i,:]) 
                labelsArray.append(temp)
                classAmounts[temp] = classAmounts[temp] + 1
        else: # use binary classification
            print("binary prediction")
            malignantClasses = [0,2,3] # the 1.,3. and 4. classes are malginant 
            classAmounts = np.zeros([2])
            weights = np.zeros([2])
            for i in range(len(labels)):
                temp = np.argmax(labels[i,:]) 
                if(temp in malignantClasses):
                    temp = 1
                else:
                    temp = 0

                labelsArray.append(temp)
                classAmounts[temp] = classAmounts[temp] + 1

        # compute classes weights, the less amount of classes -> more important is the loss function
        for i in range(len(classAmounts)):
            weights[i] = len(labels)/classAmounts[i]
    return labelsArray, weights # labelsArray == ground truth; weights == weights for the loss function

 #CNN model
 #STARTING here code from: https://github.com/sotte/pytorch_tutorial/blob/master/notebooks/03_transfer_learning.ipynb

# #not used atm
# def get_frozen(model_params):
#     return (p for p in model_params if not p.requires_grad)


# def all_trainable(model_params):
#     return all(p.requires_grad for p in model_params)
# def all_frozen(model_params):
#     return all(not p.requires_grad for p in model_params)

#define helper functions that are used
def get_trainable(model_params):
    return (p for p in model_params if p.requires_grad)

def freeze_all(model_params):
    for param in model_params:
        param.requires_grad = False

#model definition
def get_model(n_classes):
    if(_numModel == 0): # not working now
        print("not working right now")
        model = models.resnet34(pretrained=True)
        freeze_all(model.parameters())
        print(model)
        model.fc = nn.Linear(512, n_classes, bias=True) 
    if(_numModel == 1):
        #use  densenet module
        model = models.densenet121(pretrained=True)
        #freeze all weights of the current densenet121 -> as the pretrained is loaded
        #freeze_all(model.parameters())
        #replace the last layer (classifier) -> only these weights are trained
        model.classifier = nn.Linear(1024, n_classes, bias=True)    
    #DEBUG: print model (to see what the last layer is now, only works for short modules)
    print(model)
    #DEBUG END
    return model
#END CODE FROM https://github.com/sotte/pytorch_tutorial/blob/master/notebooks/03_transfer_learning.ipynb
    
# Custom data loader
class ISICDataset(Dataset):
    def __init__(self, imagePath, trainOrTest, labelsArray ):
        #declare lists
        self.im_paths = []
        labels_list = []
        testData = []
        testImages = []
        trainingImages = []
        testLabels = []
        trainLabels = []
        #get the labels of images in the validation set
        with open((imagePath + '/val_split_info.csv')) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                testData.append(row[0])
        #split images in validation set and training set
        i = 0
        for image in os.listdir(imagePath):
            if image.endswith('.jpg') or image.endswith('.JPG'):
                if image in testData:
                    #testImages.append((imagePath + "/" + image))
                    testImages.append( image)
                    testLabels.append(np.int(labelsArray[i]))
                else:
                    #trainingImages.append([imagePath + "/" + image])
                    trainingImages.append( image)
                    trainLabels.append(np.int(labelsArray[i]))
                i = i + 1
        if(trainOrTest == 0): #trainOrTest == 0 -> training set should be loaded; else test set should be loaded -> same class is used for
            #validation and test set
            labels_list = np.asarray(trainLabels) 
            self.im_paths.extend(trainingImages)
        else:
            labels_list = np.asarray(testLabels) 
            self.im_paths.extend(testImages)
        # Labels to matrix
        self.labels = labels_list 

        # Preprocessing & Data augmentation
        if(trainOrTest == 0):
            self.composed = transforms.Compose([
                    transforms.Scale(256), #arbitrary choosen 256 -as used by: https://github.com/sotte/pytorch_tutorial/blob/master/notebooks/03_transfer_learning.ipynb
                    transforms.CenterCrop(244), # for densenet image size needs to be 244 (for the other modules too, but is not working -> why)
                    transforms.ColorJitter(.3, .3, .3), #arbitrary choosen -> as used by https://github.com/sotte/pytorch_tutorial/blob/master/notebooks/03_transfer_learning.ipynb 
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(), #needed a tensor
                    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))]) # values given by: pytorch and https://github.com/sotte/pytorch_tutorial/blob/master/notebooks/03_transfer_learning.ipynb
        else:
            self.composed = transforms.Compose([
                    transforms. transforms.Scale(256), #arbitrary choosen 256
                    transforms.CenterCrop(244),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        # Get image
        x1 = Image.open(self.im_paths[idx])
        # Get label
        y = self.labels[idx]
        # Preprocessing & data augmentation
        x = self.composed(x1)
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
        #only adjust the weights of the last layer
        #with torch.set_grad_enabled(True): 
        # Calculate outputs
        output = model(data)
        # Calculate loss
        loss = loss_function(output, target)
        # Backpropagate loss
        loss.backward()
        # Apply gradients
        optimizer.step()
        if batch_idx % 50 == 0:
            print("Train Epoch:",epoch,"Loss:",loss.item(),"[", batch_idx*_batchSize,"/7500]")

            
# Testing function
def test(model, device, test_loader, loss_function):
    # Set to evaluation mode
    model.eval()
    with torch.no_grad():
        #for i, (data, target) in enumerate(test_loader):
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            #output = output.cpu()
            curr_loss = loss_function(output,target)
            if i==0:
                predictions = output
                targets = target.data
                loss = [curr_loss.data.cpu().numpy()]
            else:
                predictions = torch.cat((predictions,output))
                targets = torch.cat((targets,target.data))
                loss = np.concatenate((loss, [curr_loss.data.cpu().numpy()]))
    #get soft max -> properbility of all classes
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    softPredictions = np.empty([predictions.shape[0],predictions.shape[1]])
    for i in range(predictions.shape[0]):
        softPredictions[i] = softmax(predictions[i])
    
    #get maximum -> predicted class
    predictions = np.argmax(predictions,1)
    # Caluclate metrics
    accuracy = np.mean(np.equal(predictions,targets))
    conf_mat = confusion_matrix(targets,predictions)
    sensitivity = conf_mat.diagonal()/conf_mat.sum(axis=1)
    # Print metrics
    print("Test Accuracy",accuracy,"Test Sensitivity",np.mean(sensitivity),"Test loss",np.mean(loss))
    print("Test Sensitivity",np.mean(sensitivity),"Test loss",np.mean(loss))
    return softPredictions, np.mean(sensitivity)

#compute the softmax
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#save checkpoint
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


##### start the main program #####
if __name__ == "__main__":
    # Get datasets
    
    labelsArray, weights = np.asarray(getGroundTruth('./Includes/ISIC2018_Task3_Training_GroundTruth.csv'))
    os.chdir(_pathTrain)
    train_dataset = ISICDataset(_pathTrain, 0, labelsArray)
    train_loader = DataLoader(train_dataset,batch_size=_batchSize,shuffle=True,pin_memory=False)
    test_dataset = ISICDataset(_pathTrain, 1, labelsArray)
    test_loader = DataLoader(test_dataset,batch_size=_batchSize,shuffle=False,pin_memory=False)
    # Define device
    #device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Loss function with weights
    #loss_function = nn.CrossEntropyLoss(weight = torch.FloatTensor(weights).to(device))
    # # Loss function without weights
    loss_function = nn.CrossEntropyLoss()
    if(_bin == 0):
        model = get_model(7).to(device)
    else:
        model = get_model(2).to(device)

    optimizer = optim.Adam( get_trainable(model.parameters()), lr = _lr)
    best_pred = []
    if(_onlyEval == 0):
        #start training / evaluation loop
        for epoch in range(_numEpos):
            #Sets the learning rate to the initial LR decayed by 10 every 10 epochs, from https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
            #lr = _lr * (0.1 ** (epoch // 10))         
            #for param_group in optimizer.param_groups:
            #    param_group['lr'] = lr
            
            train(model, device, train_loader, optimizer, loss_function, epoch)
            predictions, pred = test(model, device, test_loader, loss_function)

            #still not sure how to use this, remember best prec@1 and save checkpoint
            is_best = pred > best_pred
            best_pred= max(pred, best_pred)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_pred,
            }, is_best)    
    else:
        ##### STILL AN ERROR while loading the models parameter #####
            checkpoint = torch.load('checkpoint.pth')
            # args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            predictions, pred = test(model, device, test_loader, loss_function)


    #save Predictions
    output = []
    output_bin = []
    if(_bin == 0):
        output.append('image,MEL,NV,BCC,AKIEC,BKL,DF,VASC')
        #get rid of .jpg
        for i in range(len(test_dataset.im_paths)):
            if test_dataset.im_paths[i].endswith('.jpg'):
                test_dataset.im_paths[i] =test_dataset.im_paths[i][:-4]

        for i in range(len(predictions)):
            output.append(test_dataset.im_paths[i] + 
                ',' + str(predictions[i,0]) + 
                ',' + str(predictions[i,1]) + 
                ',' + str(predictions[i,2]) + 
                ',' + str(predictions[i,3]) + 
                ',' + str(predictions[i,4]) + 
                ',' + str(predictions[i,5]) + 
                ',' + str(predictions[i,6]))
            np.savetxt('y_pred.csv',output,fmt='%s')
    else:
        output_bin.append('image,BEN,MAL')
        #get rid of .jpg
        for i in range(len(test_dataset.im_paths)):
            if test_dataset.im_paths[i].endswith('.jpg'):
                test_dataset.im_paths[i] =test_dataset.im_paths[i][:-4]

        for i in range(len(predictions)):
            output_bin.append(test_dataset.im_paths[i] + 
                ',' + str(predictions[i,0]) + 
                ',' + str(predictions[i,1]))
        np.savetxt('y_pred_bin.csv',output_bin,fmt='%s')