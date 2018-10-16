import torch
from astropy.io import fits
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np

class psf_dataset(Dataset):

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
        image = np.stack((sample_hdu[1].data, sample_hdu[2].data)).astype(np.float32)

        # Target
        zernike = sample_hdu[0].data.astype(np.float32)

        sample = {'zernike': zernike, 'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Normalize(object):
    """
        Normalize images
        with respect to unaberrated psf.
    """
    def __init__(self, root_dir):

        self.root_dir = root_dir
        self.reference_name = self.root_dir + 'psf_reference.fits'
        self.reference_hdu = fits.open(self.reference_name)
        self.norm = np.amax(self.reference_hdu[1].data)

    def __call__(self, sample):
        zernike, image = sample['zernike'], sample['image']

        image = image / self.norm

        return {'zernike': zernike, 'image': image}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
    """
    def __call__(self, sample):
        zernike, image = sample['zernike'], sample['image']

        return {'zernike': torch.from_numpy(zernike), 'image': torch.from_numpy(image)}


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

if __name__ == "__main__":

    from torchvision import transforms
    import matplotlib.pyplot as plt

    data_dir = 'dataset/'
    dataset_size = 100

    # Visualize dataset
    dataset = psf_dataset(
        root_dir=data_dir,
        size=dataset_size,
        transform=transforms.Compose([Normalize(data_dir),ToTensor()])
    )

    split = [0.9, 0.1]
    batch_size = 4
    random_seed = 42

    train_dataloader, val_dataloader = splitDataLoader(dataset,
                                                      split=[0.9, 0.1],
                                                      batch_size=batch_size,
                                                      random_seed=random_seed)
    print('Train set size: %i | Validation set size: %i' % (4*len(train_dataloader),
                                                                   4*len(val_dataloader)))

    id = 53
    sample = dataset[id]
    image_in = sample['image'][0]
    image_out = sample['image'][1]

    reference_name = 'dataset/psf_reference.fits'
    reference_hdu = fits.open(reference_name)
    norm = np.amax(reference_hdu[1].data)

    f, axarr = plt.subplots(1, 3, figsize=(10, 10))
    im1 = axarr[0].imshow(reference_hdu[1].data / norm)
    axarr[0].set_title("Reference")
    plt.colorbar(im1, ax = axarr[0], fraction=0.046)
    im2 = axarr[1].imshow(image_in)
    axarr[1].set_title("In")
    plt.colorbar(im2, ax = axarr[1], fraction=0.046)
    im3 = axarr[2].imshow(image_out)
    axarr[2].set_title("Out")
    plt.colorbar(im3, ax = axarr[2], fraction=0.046)
    plt.show()
