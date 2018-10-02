import numpy
import numpy as np
from matplotlib import pyplot
from matplotlib import pyplot as plt
#from tqdm import tqdm, tnrange
from astropy.io import fits
from scipy.io import loadmat, savemat
import time

import aotools
from aotools.turbulence import infinitephasescreen, phasescreen
from soapy import SCI, confParse

import proper

gridsize = 128
wavelength = 2.2e-6
# Nb of Zernike to use in the generation
nbOfZernike = 40
# Nb of PSFs to generate
nbOfPsfs = 100
# Telescope diameter
diameter = 1
pixelScale = 0.11 # [''/px]

# -- Generate set of random coefficients --
# uniformely distributed from [-1, 1]
coeffs = np.random.random((nbOfPsfs, nbOfZernike))*2 - 1

# Zernike mode index -- we will start from mode Z2=tip
coeffIdx = np.arange(2, nbOfZernike+2)

# normalize to a given total wfs rms
normInNmRms = 2000
coeffs = np.array([coeffs[k, :] / np.abs(coeffs[k,:]).sum() *
                   normInNmRms for k in range(nbOfPsfs)])


SOAPY_CONF = "conf/psf.yaml"
config = confParse.loadSoapyConfig(SOAPY_CONF)

# Set the scientific camera field-of-view to
# match the desired pixel scale
config.scis[0].pxlScale = pixelScale

config.calcParams()

# Assert configuration is similar for SOAPY and PROPER
assert config.sim.pupilSize == gridsize
assert config.scis[0].wavelength == wavelength
assert config.scis[0].pxls == gridsize

# Pupil mask
mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

# Create the PSF image
psfObj = SCI.PSF(config, nSci=0, mask=mask)

# Soapy camera pixel scale
pxscale_soapy = psfObj.fov / psfObj.nx_pixels
print("Pixel scale is {0:1f} ''/pixel".format(pxscale_soapy))
print("FoV is {0:1f} ''".format(config.scis[0].FOV))

# Generate the Zernike modal basis
Zs = aotools.zernikeArray(nbOfZernike + 1, config.sim.pupilSize, norm='rms')
#Zs = np.array([np.pad(Zs[k,:,:], config.sim.simPad, 'constant', constant_values=0) for k in range(nbOfZernike+1)])

# Illustration of Zernike basis
# Astigmatism
plt.figure()
plt.imshow(Zs[4,:,:])

# Illustration of an input phase screen
tt = np.squeeze(np.sum(coeffs[-1, :, None, None] * Zs[1:, :,:], axis=0))
plt.figure()
plt.imshow(tt)

psfs_soapy = np.zeros((nbOfPsfs, psfObj.detector.shape[0], psfObj.detector.shape[1]))

psfObj.frame(Zs[0,:,:])
t0 = time.time()

for i in range(nbOfPsfs):
    #print(i)
    aberrations = np.squeeze(np.sum(coeffs[i, :, None, None] * Zs[1:, :, :], axis=0))
    psfObj.frame(aberrations)
    #print('>>')
    psfs_soapy[i, :, :] = np.copy(psfObj.frame(aberrations))
t_soapy = time.time() - t0
print('Loop duration {0:2f}sec'.format(t_soapy))

plt.figure()
plt.imshow(psfObj.los.phase)
plt.figure()
plt.imshow(psfObj.mask)
plt.show()