import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataset import PSFDataset, ToTensor, Normalize
from net import Net


dataset = PSFDataset(root_dir='psfs/',
                     size=500,
                     transform=transforms.Compose([
                                    Normalize(),
                                    ToTensor()]))

# Dataset split 70/15/15
train_size = int(0.7 * len(dataset))                        # Train
val_size = int(0.15 * len(dataset))                         # Tune hyperparameters
test_size = len(dataset) - train_size - val_size            # Unbiased evaluation

# Batch size
batch_size = 4

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Net
net = Net()
# Number of epochs
n_epochs = 5
# Number of batches
n_batches = len(train_dataloader)
# Learning rate
learning_rate = 0.01
# Loss function
criterion = nn.MSELoss()
# Optimizer
optimizer = optim.Adam(net.parameters(), lr =learning_rate)
#Time for printing
training_start_time = time.time()

for epoch in range(n_epochs):

    running_loss = 0.0
    print_every = n_batches // 10
    start_time = time.time()

    for i_batch, sample_batched in enumerate(train_dataloader):

        zernike = sample_batched['zernike']
        image = sample_batched['image']

        # Set the parameter gradients to zero
        optimizer.zero_grad()

        # Forward pass, backward pass, optimize
        outputs = net(image)
        loss = criterion(outputs, zernike)
        #print(outputs)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss
        if (i_batch + 1) % (print_every + 1) == 0:
            print('[%d, %5d] loss: %.3f time: %.3f s' %
                  (epoch + 1, batch_size*(i_batch + 1), running_loss / print_every,  time.time() - start_time))
            running_loss = 0.0
            start_time = time.time()

    # At the end of the epoch, do a pass on the validation set
    val_loss = 0
    for i_batch, sample_batched in enumerate(val_dataloader):

        zernike = sample_batched['zernike']
        image = sample_batched['image']

        outputs = net(image)
        loss = criterion(outputs, zernike)
        val_loss += loss
    print('Validation loss: %.3f ' % (val_loss / len(val_dataloader)))

    print('Training finished in %.3f s' % (time.time() - start_time))
    
