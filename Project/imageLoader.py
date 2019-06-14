import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#import torchvision.datasets as datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class ImageLoader():
    def __init__(self,
                 batch_size=32,
                 shuffle=True,
                 n_workers=0,
                 ):
        self.batch_size=batch_size
        self.n_workers = n_workers
        self.transformations = transforms.Compose([
                transforms.ToTensor(),
                ])

    def loadData(self,
                 path
                 ):
        if path[-1] != "/":
            path += "/"
        train_path = path + "train"
        valid_path = path + "valid"
        training_set = datasets.ImageFolder(train_path, 
                                            transform=self.transformations)
        validation_set = datasets.ImageFolder(valid_path, 
                                              transform=self.transformations)
        training_loader = DataLoader(training_set,
                                     batch_size=self.batch_size,
                                     num_workers=self.n_workers,
                                     shuffle=self.shuffle)
        validation_loader = DataLoader(training_set,
                                       batch_size=self.batch_size,
                                       num_workers=self.n_workers,
                                       shuffle=self.shuffle)
        return training_loader, validation_loader
