import numpy as np
import torch
from torch.utils.data import Dataset

class UTKDataset(Dataset):
    '''
        Inputs:
            dataFrame : Pandas dataFrame
            transform : The transform to apply to the dataset
    '''
    def __init__(self, dataFrame, label_name="gender", transform=None):
        # read in the transforms
        self.transform = transform
        
        # Use the dataFrame to get the pixel values
        data_holder = dataFrame.pixels.apply(lambda x: np.array(x.split(" "),dtype=float))
        arr = np.stack(data_holder)
        arr = arr / 255.0
        arr = arr.astype('float32')
        arr = arr.reshape(arr.shape[0], 48, 48, 1)
        # reshape into 48x48x1
        self.data = arr
        
        # get the gender, and ethnicity label arrays
        # self.gender_label = np.array(dataFrame.gender[:])
        # self.eth_label = np.array(dataFrame.ethnicity[:])
        if label_name == "gender":
            self.targets = np.array(dataFrame.gender[:])
            self.groups = np.array(dataFrame.ethnicity[:])
            # self.groups = np.where(dataFrame.ethnicity[:] == 0, 0, 1)
        else:
            self.targets = np.array(dataFrame.ethnicity[:])
            self.groups = np.array(dataFrame.gender[:])
    
    # override the length function
    def __len__(self):
        return len(self.data)
    
    # override the getitem function
    def __getitem__(self, index):
        # load the data at index and apply transform
        data = self.data[index]
        data = self.transform(data)
        
        # load the labels into a list and convert to tensors
        target = torch.tensor(self.targets[index])
        group = torch.tensor(self.groups[index])
        
        # return data labels
        return data, target, group
     