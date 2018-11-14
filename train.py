import os
import time
import utils
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from dataset import psf_dataset, splitDataLoader, ToTensor, Normalize
from utils_visdom import VisdomWebServer
import aotools

def train(model, dataset, optimizer, criterion, split=[0.9, 0.1], batch_size=32, 
          n_epochs=1, model_dir='./', random_seed=None, visdom=False):
    
    # Create directory if doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Logging
    log_path = os.path.join(model_dir, 'logs.log')
    utils.set_logger(log_path)
    
    # Visdom support
    if visdom:
        vis = VisdomWebServer()
        
    # Reduce lr by 10 every 10 epoch without improvement
    scheduler = None # optim.lr_scheduler.StepLR(optimizer, step_size=350, gamma=0.1) 
        
    # Metrics
    metrics_path = os.path.join(model_dir, 'metrics.json')
    
    metrics = {
        'model': model_dir,
        'optimizer': optimizer.__class__.__name__,
        'criterion': criterion.__class__.__name__,
        'scheduler': scheduler.__class__.__name__,
        'dataset_size': int(len(dataset)),
        'train_size': int(split[0]*len(dataset)),
        'test_size': int(split[1]*len(dataset)),
        'n_epoch': n_epochs,
        'batch_size': batch_size,
        'learning_rate': get_lr(optimizer),
        'train_loss': [],
        'val_loss': [],
        'zernike_train_loss': [],
        'zernike_val_loss': []
    }
    
    # Zernike basis
    z_basis = torch.as_tensor(aotools.zernikeArray(21, 128, norm='rms'), dtype=torch.float32)
    
    # Dataset
    dataloaders = {}
    dataloaders['train'], dataloaders['val'] = splitDataLoader(dataset, split=split, 
                                                             batch_size=batch_size, random_seed=random_seed)
    
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #logging.info('Training started on %s' % (device))
    #if torch.cuda.device_count() > 1:
    #    model = nn.DataParallel(model)
    #    model.to(device)  
   
    # Training
    since = time.time()
    dataset_size = {
        'train':int(split[0]*len(dataset)),
        'val':int(split[1]*len(dataset))
    }
    best_loss = 0.0
    
    for epoch in range(n_epochs):
        
        logging.info('-'*30)
        epoch_time = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            zernike_loss =0.0
            
            for _, sample in enumerate(dataloaders[phase]):
                # GPU support 
                inputs = sample['image'].to(device)
                z_coeffs_0 = sample['zernike'].to(device)
                z_basis_0 = z_basis.to(device)
                phase_0 = get_phase(z_coeffs_0, z_basis_0)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward: track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Network return phase and zernike coeffs
                    phase_, z_coeffs = model(inputs)
                    loss = criterion(phase_, phase_0)
                    z_loss = criterion(z_coeffs, z_coeffs_0)
                    
                    # backward
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()   
                     
                    # Dropout and BatchNorm support
                    #model.eval()
                    #phase_, z_coeffs = model(inputs)
                    #loss = criterion(phase_, phase_0)
                    #z_loss = criterion(z_coeffs, z_coeffs_0)
                    #model.train()
                    
                    zernike_loss += z_loss.item() * inputs.size(0)  
                    running_loss += loss.item() * inputs.size(0)
                    
            logging.info('[%i/%i] %s loss: %f' % (epoch+1, n_epochs, phase, running_loss / dataset_size[phase]))
            
            # Update metrics
            metrics[phase+'_loss'].append(running_loss / dataset_size[phase])
            metrics['zernike_'+phase+'_loss'].append(zernike_loss / dataset_size[phase])
                
            # Adaptive learning rate
            if phase == 'val':
                #scheduler.step()
                # Save weigths
                if epoch == 0 or running_loss < best_loss:
                    best_loss = running_loss
                    model_path = os.path.join(model_dir, 'model.pth')
                    torch.save(model.state_dict(), model_path)
                # Save metrics
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4) 
                # Visdom update 
                if visdom:
                    vis.update(metrics)
                    
        logging.info('[%i/%i] Time: %f s' % (epoch + 1, n_epochs, time.time()-epoch_time))
        
    time_elapsed = time.time() - since    
    logging.info('[-----] All epochs completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

            
        
def get_lr(optimizer):
    for p in optimizer.param_groups:
        lr = p['lr']
    return lr  


def get_phase(z_coeffs, z_basis):
    z_phase = z_coeffs[:,:, None, None] * z_basis[None, 1:,:,:]
    total_phase = torch.sum(z_phase, dim=1)
    return total_phase
