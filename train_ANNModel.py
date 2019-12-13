""" Train CNN model to Classify Satellite Image Data from Sat-4 Dataset; RGB channel"""

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
if not train_on_gpu:
    print('CUDA is not available. Training on CPU.........')
else:
    print('CUDA is available. Training on GPU.....')
    
#helper funtion for decode One-Hot
def decode_landmarks(array):
    """One-hot Decde"""
    np.flip(array)    
    return np.argmax(array)
    

#Helper function to convert RGB to Gray
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
    
    
#Custom DataLoader for sate-4 image dataset
    
class SatFourDataset(Dataset):
    """Satellite 4 Aerial Image Datatset"""
    
    def __init__(self,mat_file,transform=None):
        """
            Arg: MATfile containing training and test data along with ground truth
        """
        
        self.landdata=loadmat(mat_file)
        self.transform=transform
        
        
    def __len__(self):
        x=self.landdata['train_x']
        data_len=x.shape[-1]
        return data_len
        
        
    def __getitem__(self,idx):
    
        x=self.landdata['train_x']
        image=x[:,:,:3,idx]
        #image=transforms.ToPILImage(image)
        if self.transform:
            image=self.transform(image)
        y=self.landdata['train_y']
        landmark=y[:,idx]
        landmark_type=decode_landmarks(landmark)
        sample = {'image': image, 'landmarks': landmark_type}
       
        return sample


#Transform dataset
transformed_train_dataset=SatFourDataset(mat_file='sat-4-full.mat',transform=transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]))
print('transformed_train_dataset',len(transformed_train_dataset))


#visualize if transformation is correct
"""for i in range(2):
    sample = transformed_train_dataset[i]
    image=sample['image']
    print(image)"""


#Split Training and Validation
valid_size=0.2 # number of subprocesses to use for data loading
num_train = len(transformed_train_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

num_workers=0   # number of subprocesses to use for data loading
batch_size=20  # samples per batch to load


# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = DataLoader(transformed_train_dataset, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers)
valid_loader = DataLoader(transformed_train_dataset, batch_size=batch_size, 
    sampler=valid_sampler, num_workers=num_workers)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
  
    
classes=['Barren-Land','Trees','Grass-Land','Other'] #Classify according to name

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 512
        hidden_2 = 256
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 4)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        return x

# initialize the NN
model = Net()
print(model)

# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()
    

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# number of epochs to train the model
n_epochs = 30
since = time.time()
model.train()

valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    model.train()
    for i_batch, sample_batched in enumerate(train_loader):
        if train_on_gpu:
            data=sample_batched['image']
            target=sample_batched['landmarks']
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
                 
         model.eval()
    for i_batch,sample_batched in enumerate(valid_loader):
        if train_on_gpu:
            data=sample_batched['image']
            target=sample_batched['landmarks']
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
 
            valid_loss  += loss.item()*data.size(0)
    # calculate average losses
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_Sat4.pt')
        valid_loss_min = valid_loss
        
time_elapsed = time.time() - since
print('Total Training time,',time_elapsed)