""" CNN model to Classify Satellite Image Data from Sat-4 Dataset; RGB-NIR channel"""

import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
from scipy.io import loadmat
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


#check on CUDA
train_on_gpu=torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")

#helper funtion for decode One-Hot
def decode_landmarks(array):
    """One-hot Decde"""
    np.flip(array)    
    return np.argmax(array)
    


class SatFourDatasetIR(Dataset):
    """Satellite 4 Aerial Image Datatset"""
    
    def __init__(self,mat_file,transform=None):
        """
            Arg: MATfile containing training and test data along with ground truth
        """
        
        self.landdata=loadmat(mat_file)
        self.transform=transform
        
        
    def __len__(self):
        x=self.landdata['test_x']
        data_len=x.shape[-1]
        return data_len
        
        
    def __getitem__(self,idx):
    
        x=self.landdata['test_x']
        image_ir=x[:,:,-1,idx]
        image=x[:,:,:3,idx]
        if self.transform:
            image_rgb=self.transform(image)
            image_ir=self.transform(image_ir)
        rgb_transform=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image=torch.cat((image_rgb, image_ir), 0)
        y=self.landdata['test_y']
        landmark=y[:,idx]
        landmark_type=decode_landmarks(landmark)
        sample = {'image': image, 'landmarks': landmark_type}
       
        return sample



#Load Test data
transformed_test_dataset=SatFourDatasetIR(mat_file='sat-4-full.mat',transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(transformed_test_dataset, batch_size=20, num_workers=0)



#loading the architecture
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 3,padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*7*7,1000)
        self.fc2 = nn.Linear(1000,400)
        self.fc3 = nn.Linear(400,4)
        self.dropout = nn.Dropout(0.25)
        
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x= F.relu(self.conv3(x))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        
        return x
        
        
# create a complete CNN
model = Model()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

#Load the model with lowest validation loss
model.load_state_dict(torch.load('model_IRSat4.pt'))
  
    
classes=['Barren-Land','Trees','Grass-Land','Other'] #Classify according to name
# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(4))
class_total = list(0. for i in range(4))
batch_size=20
model.eval()
criterion = nn.CrossEntropyLoss()
for i_batch,sample_batched in enumerate(test_loader):

    if train_on_gpu:
        data=sample_batched['image']
        target=sample_batched['landmarks']
        data, target = data.cuda(), target.cuda()

    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    _, pred = torch.max(output, 1)    
    all_pred=np.squeeze(pred.numpy()) if not train_on_gpu else np.squeeze(pred.cpu().numpy())
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
        
    #Create confusion Matrix
    conf_matrix = torch.zeros(4, 4)
    for t, p in zip(target, all_pred):
        conf_matrix[t, p] += 1

   

TP = conf_matrix.diag()
for c in range(4):
    idx = torch.ones(nb_classes).byte()
    idx[c] = 0
    # all non-class samples classified as non-class
    TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum() 
    # all non-class samples classified as class
    FP = conf_matrix[idx, c].sum()
    # all class samples not classified as class
    FN = conf_matrix[c, idx].sum()
    
    print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
        c, TP[c], TN, FP, FN))

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(4):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2f%% (%2f/%2f)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2f%% (%2f/%2f)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
