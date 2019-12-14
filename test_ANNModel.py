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
    


class SatFourDataset(Dataset):
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
        image=x[:,:,:3,idx]
        #image=transforms.ToPILImage(image)
        if self.transform:
            image=self.transform(image)
        y=self.landdata['test_y']
        landmark=y[:,idx]
        landmark_type=decode_landmarks(landmark)
        sample = {'image': image, 'landmarks': landmark_type}
       
        return sample


#Load Test data
transformed_test_dataset=SatFourDataset(mat_file='sat-4-full.mat',transform=transforms.Compose([transforms.ToPILImage(),transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(transformed_test_dataset, batch_size=20, num_workers=0)


#Trained model architecture
class ANN_model(nn.Module):
    def __init__(self):
        super(ANN_model, self).__init__()
        # number of hidden nodes in first layer (512), second layer 256
        hidden_1 = 512
        hidden_2 = 256
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 4)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 28 * 28) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # add output layer
        return x

model = ANN_model()
print(model)
if train_on_gpu:
    model.cuda()

#Load the model with lowest validation loss
model.load_state_dict(torch.load('model_ANNSat4.pt'))

  
    
classes=['Barren-Land','Trees','Grass-Land','Other'] #Classify according to name

# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(4))
class_total = list(0. for i in range(4))
batch_size=20
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()
#evaluation mood
model.eval()

conf_matrix = torch.zeros(4, 4)
for i_batch,sample_batched in enumerate(test_loader):
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data=sample_batched['image']
        target=sample_batched['landmarks']
        data, target = data.cuda(), target.cuda()

    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    _, pred = torch.max(output, 1)
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1
     for t, p in zip(target, all_pred):
        conf_matrix[t, p] += 1

print(conf_matrix)
# calculate and print avg test loss
test_loss = test_loss/len(test_loader.sampler)
print('Test Loss: {:.6f}\n'.format(test_loss))

TP = conf_matrix.diag()
for c in range(4):
    idx = torch.ones(4).byte()
    idx[c] = 0
    # all non-class samples classified as non-class
    TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum() 
    # all non-class samples classified as class
    FP = conf_matrix[idx, c].sum()
    # all class samples not classified as class
    FN = conf_matrix[c, idx].sum()
    
    print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
        c, TP[c], TN, FP, FN))
# calculate and print avg test loss
test_loss = test_loss/len(test_loader.sampler)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(4):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))