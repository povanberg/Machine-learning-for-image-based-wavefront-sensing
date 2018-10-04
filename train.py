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
    json_path = os.path.join(args.model_dir, 'params.json')
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
    # Loss function
    criterion = nn.MSELoss()
    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    log_path = os.path.join(args.model_dir, 'logs.log')
    utils.set_logger(log_path)

    start_time = time.time()
    for epoch in range(params.num_epochs):

        running_loss = 0.0
        log_every = len(train_dataloader) // 10
        epoch_time = time.time()

        # Training
        net.train()
        for i_batch, sample_batched in enumerate(train_dataloader):

            zernike = sample_batched['zernike']
            image = sample_batched['image']

            # Forward pass, backward pass, optimize
            outputs = net(image)
            loss = criterion(outputs, zernike)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss)
            # Print statistics
            if (i_batch + 1) % (log_every) == 0:
                logging.info('[%d, %5d] loss: %.3f time: %.3f s' %
                      (epoch + 1, params.batch_size * (i_batch + 1), running_loss / log_every, time.time() - epoch_time))
                running_loss = 0.0
                epoch_time = time.time()

        # At the end of the epoch, do a pass on the validation set
        net.eval()
        val_loss = 0.0
        for i_batch, sample_batched in enumerate(val_dataloader):

            zernike = sample_batched['zernike']
            image = sample_batched['image']

            outputs = net(image)
            loss = criterion(outputs, zernike)
            val_loss += float(loss)

        # Save best val metrics in a json file in the model directory
        accuracy = val_loss / len(val_dataloader)
        metrics_json_path = os.path.join(args.model_dir, "metrics.json")
        metrics = utils.Params(metrics_json_path)
        if not metrics.hasKey(metrics_json_path, 'accuracy') or metrics.accuracy > accuracy:
            metrics.accuracy = accuracy
            metrics.save(metrics_json_path)
            checkpoint_path = os.path.join(args.model_dir, 'checkpoint.pth')
            torch.save(net.state_dict(), checkpoint_path)
        
        logging.info('Validation loss: %.3f ' % (val_loss / len(val_dataloader)))

    logging.info('Training finished in %.3f s' % (time.time() - start_time))


