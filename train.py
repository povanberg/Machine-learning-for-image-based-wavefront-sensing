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


def train(model, dataset, optimizer, criterion, split=[0.9, 0.1],
          batch_size=32, n_epoch=1, model_dir='./', random_seed=None, visdom=False, decay=False):

    # Prepare dataset
    train_dataloader, val_dataloader = splitDataLoader(
                                                        dataset,
                                                        split=split,
                                                        batch_size=batch_size,
                                                        random_seed=random_seed
                                                       )
    

    # Logging
    log_path = os.path.join(model_dir, 'logs.log')
    utils.set_logger(log_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info('Training started on %s' % (device))

    # Visdom
    if visdom:
        vis = VisdomWebServer()

    # Metrics
    metrics_path = os.path.join(model_dir, 'metrics.json')
    for p in optimizer.param_groups:
        lr = p['lr']
    metrics = {
        'model': model_dir,
        'optimizer': optimizer.__class__.__name__,
        'criterion': criterion.__class__.__name__,
        'dataset_size': len(dataset),
        'train_size': split[0]*len(dataset),
        'test_size': split[1]*len(dataset),
        'n_epoch': n_epoch,
        'batch_size': batch_size,
        'learning_rate': lr,
        'train_loss': [],
        'val_loss': []
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    min_loss = 0.0
    train_time = time.time()

    # Start training
    for epoch in range(n_epoch):

        train_loss = 0.0
        epoch_time = time.time()
        
        if decay:
            lr_new = adjust_learning_rate(optimizer, epoch, lr)
            if lr_new != lr:
                logging.info('Learning rate updated: %f ' % (lr_new))

        for i_batch, sample_batched in enumerate(train_dataloader):

            model.train()
            zernike = sample_batched['zernike'].to(device)
            image = sample_batched['image'].to(device)

            optimizer.zero_grad()
            
            estimated_zernike, aux = model(image)
            if isinstance(estimated_zernike, tuple):
                loss = sum((criterion(o,zernike) for o in estimated_zernike))
            else:
                loss = criterion(estimated_zernike, zernike)
            #loss = criterion(estimated_zernike, zernike)
            loss.backward()
            optimizer.step()

            #model.eval()
            #estimated_zernike = model(image)
            #loss = criterion(estimated_zernike, zernike)
            train_loss += float(loss) / len(train_dataloader)  
        
        logging.info('[%i/%i] Train loss: %f ' % (epoch+1, n_epoch, train_loss))

        val_loss = eval(model,
                        dataloader=val_dataloader,
                        criterion=criterion,
                        device=device)

        logging.info('[%i/%i] Validation loss: %f ' % (epoch+1, n_epoch, val_loss))

        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)

        # Save model and metrics
        if epoch == 0 or val_loss < min_loss:
            min_loss = val_loss
            model_path = os.path.join(model_dir, 'model.pth')
            torch.save(model.state_dict(), model_path)

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        if visdom:
            vis.update(metrics)

        logging.info('[%i/%i] Time: %f s' % (epoch + 1, n_epoch, time.time()-epoch_time))
        logging.info('-'*30)

    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info('All epochs completed in %f s' % (time.time() - train_time))

def adjust_learning_rate(optimizer, epoch, lr):
    lr_new = lr * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new   
    return lr_new    
    
def eval(model, dataloader, criterion, device):

    model.eval()
    val_loss = 0.0

    for i_batch, sample_batched in enumerate(dataloader):
        zernike = sample_batched['zernike'].to(device)
        image = sample_batched['image'].to(device)

        estimated_zernike = model(image)
        loss = criterion(estimated_zernike, zernike)

        val_loss += float(loss) / len(dataloader)

    return val_loss

if __name__ == "__main__":

    from utils import get_metrics, plot_learningcurve

    print('Training simple model')
    print('-'*22)

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()

            self.conv1 = nn.Conv2d(2, 20, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(40, 20, kernel_size=1, stride=1, padding=0)

            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            self.fc1 = torch.nn.Linear(16 * 16 * 20, 256)
            self.fc2 = torch.nn.Linear(256, 20)

        def forward(self, x):
            # 128x128x2
            x = F.relu(self.conv1(x))       # 128x128x20
            x = self.pool(x)                # 64x64x20
            x = F.relu(self.conv2(x))       # 64x64x40
            x = self.pool(x)                # 32x32x40
            x = F.relu(self.conv3(x))       # 32x32x20
            x = self.pool(x)                # 16x16x20
            x = x.view(-1, 20 * 16 * 16)    # 1x5120      (Reshape for fully connected layer)
            x = F.relu(self.fc1(x))         # 1x256
            x = self.fc2(x)                 # 1x20                     (Zernike coeffs)
            return x

    model = Net()

    # GPU support
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Dataset
    data_dir = 'dataset/'
    dataset_size = 10
    dataset = psf_dataset(
        root_dir=data_dir,
        size=dataset_size,
        transform=transforms.Compose([Normalize(data_dir),ToTensor()])
    )

    # Reproducibility
    random_seed = 42

    train(model, dataset, optimizer, criterion,
          split=[0.9, 0.1],
          batch_size=32,
          n_epoch=20,
          random_seed=42,
          model_dir='model_test/')

    metrics = get_metrics('experiments/example')
    plot_learningcurve(metrics)
