import torch
from astropy.io import fits
from torch.utils.data import Dataset
import numpy as np

class PSFDataset(Dataset):

    def __init__(self, root_dir, size, transform=None):
        """
        Args:
            size (int): Number of fits files
            root_dir (string): Path to fits files
        """
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
        # Stack in and out of focus images (128,128,2)
        image = np.stack((sample_hdu[1].data, sample_hdu[2].data))
        # Row to Column vector
        zernike = sample_hdu[0].data.astype(np.float64)
        sample = {'zernike': zernike, 'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Normalize(object):
    """
        Normalize images
    """
    def __call__(self, sample):
        zernike, image = sample['zernike'], sample['image']

        for i  in range(2):
            mean = np.mean(image[i,:,:])
            std = np.std(image[i,:,:])
            image[i,:,:] = (image[i,:,:] - mean) / std

        return {'zernike': zernike, 'image': image}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        zernike, image = sample['zernike'], sample['image']

        return {'zernike': torch.from_numpy(zernike), 'image': torch.from_numpy(image)}

