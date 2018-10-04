import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset import PSFDataset, ToTensor, Normalize
from net import Net

import os
import logging
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='psfs/', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='models/', help="Directory containing params.json")

if __name__ == '__main__':

    args = parser.parse_args()

    # Load params
    json_path = os.path.join('models/baseline/lr0.001_b4_e1', 'params.json')
    assert os.path.isfile(json_path), logging.error("No json configuration file found at {}".format(json_path))
    params = utils.Params(json_path)

    # Load dataset
    dataset = PSFDataset(root_dir=args.data_dir, size=params.dataset_size,
                         transform=transforms.Compose([Normalize(), ToTensor()]))
    train_size = int(params.dev_split[0] * len(dataset))
    val_size = int(params.dev_split[1] * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=True, num_workers=4)

    # Load convolutional network
    net = Net()
    state_dict = torch.load('models/baseline/lr0.001_b4_e1/checkpoint.pth')
    net.load_state_dict(state_dict)
    # Loss function
    criterion = nn.MSELoss()
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    # At the end of the epoch, do a pass on the validation set
    net.eval()
    val_loss = 0.0
    for i_batch, sample_batched in enumerate(val_dataloader):
        zernike = sample_batched['zernike']
        image = sample_batched['image']

        outputs = net(image)
        loss = criterion(outputs, zernike)
        val_loss += float(loss)

    print('Validation loss: %.3f ' % (val_loss / len(val_dataloader)))


