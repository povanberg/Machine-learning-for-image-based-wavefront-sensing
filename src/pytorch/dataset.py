import torch
from astropy.io import fits
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from astropy.visualization import SqrtStretch, MinMaxInterval
import numpy as np

class psf_dataset(Dataset):

    def __init__(self, root_dir, size, transform=None):
        self.size = size
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, id):

        if id >= self.size:
            raise ValueError('[Dataset] Index out of bounds')
            return None

        sample_name = self.root_dir + 'psf_' + str(int(id)) + '.fits'
        sample_hdu = fits.open(sample_name)

        image = np.stack((sample_hdu[2].data, sample_hdu[3].data)).astype(np.float32)

        phase = sample_hdu[1].data.astype(np.float32)

        sample = {'phase': phase, 'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Normalize(object):
    def __call__(self, sample):
        phase, image = sample['phase'], sample['image']
        
        image[0] = minmax(np.sqrt(image[0]))
        image[1] = minmax(np.sqrt(image[1]))
        
        phase = (phase/2200.)*2*np.pi

        return {'phase': phase, 'image': image}

    
def minmax(array):
    a_min = np.min(array)
    a_max = np.max(array)
    return (array-a_min)/(a_max-a_min)    

class ToTensor(object):
    def __call__(self, sample):
        phase, image = sample['phase'], sample['image']

        return {'phase': torch.from_numpy(phase), 'image': torch.from_numpy(image)}

class Noise(object):
    def __call__(self, sample):
        phase, image = sample['phase'], sample['image']
        
        noise_intensity = 1000
        image[0] = minmax(image[0])
        image[1] = minmax(image[1])
        image[0] = np.random.poisson(lam=noise_intensity*image[0], size=None)
        image[1] = np.random.poisson(lam=noise_intensity*image[1], size=None)

        return {'phase': phase, 'image': image}


def splitDataLoader(dataset, split=[0.9, 0.1], batch_size=32, random_seed=None, shuffle=True):
    indices = list(range(len(dataset)))
    s = int(np.floor(split[1] * len(dataset)))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[s:], indices[:s]

    train_sampler, val_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=val_sampler)

    return train_dataloader, val_dataloader
