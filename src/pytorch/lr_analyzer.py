import os
import time
import utils
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import transforms
from dataset import psf_dataset, splitDataLoader, ToTensor, Normalize
from utils_visdom import VisdomWebServer
import aotools

def lr_analyzer(model, dataset, optimizer, criterion, split=[0.9, 0.1], batch_size=64, lr=[1e-5, 1e-1]):
   
    for p in optimizer.param_groups:
        p['lr'] = lr[0]

    lr_log = np.geomspace(lr[0], lr[1], 100)

    # Dataset
    dataloaders, _ = splitDataLoader(dataset, split=split, batch_size=batch_size)
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    losses = []
    lrs = []
    
    running_loss = 0.0
    it = 0
            
    for _, sample in enumerate(dataloaders):
        # GPU support 
        inputs = sample['image'].to(device)
        phase_0 = sample['phase'].to(device)
                
        # zero the parameter gradients
        optimizer.zero_grad()
                
        # forward: track history if only in train
        with torch.set_grad_enabled(True):

            # Network return phase and zernike coeffs
            phase_estimation = model(inputs)
            loss = criterion(torch.squeeze(phase_estimation), phase_0)
            loss.backward()
            optimizer.step()   

        losses.append(loss.item())
        lrs.append(get_lr(optimizer))
        
        if it == 100:
            break
            
        #update lr
        for p in optimizer.param_groups:
            p['lr'] = lr_log[it]
                    
        it +=1  
        
    return losses, lrs
        
def get_lr(optimizer):
    for p in optimizer.param_groups:
        lr = p['lr']
    return lr  
