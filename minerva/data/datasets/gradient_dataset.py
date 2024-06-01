import numpy as np
from minerva.data.datasets.supervised_dataset import SupervisedReconstructionDataset
from minerva.data.readers.reader import _Reader
from minerva.transforms.transform import _Transform
import torch
from typing import List


def generate_gradient(shape: tuple[int, int]) -> np.ndarray:              
    
    '''
    Inputs in format (H, W)
    Outputs a gradient from 0 to 1 in both x and y directions
    Channel 0 gradient on W and Channel 1 gradient on H
    '''
    
    xx, yy = np.meshgrid(np.linspace(0, 1, shape[1]), np.linspace(0, 1, shape[0]))
    gradient = np.stack([xx, yy], axis=-1)
    return gradient


class GradientDataset(SupervisedReconstructionDataset):
    
    def __init__ (self, readers: List[_Reader], transforms: _Transform | None = None):
        super().__init__(readers, transforms)
        
    def __getitem__(self, idx):
        
        data = []
        seed = np.random.randint(2147483647)
        
        for reader, transform in zip(self.readers, self.transforms):
            sample = reader[idx]
            if sample.shape[-1] == 3:
                sample = sample[:,:,0]
            gradient = generate_gradient(sample.shape)[:,:,1]
            image = np.concatenate([np.expand_dims(sample, axis=2), np.expand_dims(gradient, axis=2)], axis=2)
            np.random.seed(seed)
            if transform is not None:
                image = transform(image)
            data.append(image)
                
        return(data[0], data[1][0].to(torch.int64))
        