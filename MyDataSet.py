import os
import cv2
import numpy as np
from torchvision import transforms
import torch

# store labels as numbers
dic = {
    "CNV"     : 0,
    "DME"     : 1,
    "DRUSEN"  : 2,
    "NORMAL"  : 3
}

# Dataset loads the images from drive to RAM and as tensors
class OCTDataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # resize all images into 224x224
            transforms.Resize(size=(224, 224), antialias=True),
        ])

        self.X, self.Y = [], []

        for folder_name in ["CNV", "DME", "DRUSEN", "NORMAL"]:
            path = os.path.join(dir, folder_name)
            file_names = sorted(os.listdir(path))
            # It took around 80GB of RAM to load all the images
            
            for file_name in file_names:
                img = np.array(cv2.imread(os.path.join(path, file_name)))
                img = self.transform(img)
                self.X.append(img)
                # dic[folder_name] is the corresponding label of the image as a number
                self.Y.append(dic[folder_name])
                del img
    
        # make sure the length of X and Y is the same
        assert len(self.X) == len(self.Y)

        self.length = len(self.Y)

    def __len__(self):
      return self.length

    def __getitem__(self, idx):

      x, y = self.X[idx], self.Y[idx]

      return torch.FloatTensor(x), y
