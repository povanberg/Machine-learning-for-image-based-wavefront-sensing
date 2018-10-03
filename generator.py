import numpy as np
from matplotlib import pyplot as plt
import time

import aotools
from soapy import SCI, confParse
from astropy.io import fits

# Pixel size of science camera
gridsize = 128
# Observational wavelength
wavelength = 2.2e-6
# Nb of PSFs to generate
nbOfPsfs = 1000
# Telescope diameter
diameter = 10
# Pixel scale
pixelScale = 0.01 # [''/px]s
# Nb of Zernike to use in the generation
nbOfZernike = 20
# Zernike mode index -- we will start from mode Z2=tip
coeffIdx = np.arange(2, nbOfZernike+2)

# Soapy
SOAPY_CONF = "config/psf.yaml"
config = confParse.loadSoapyConfig(SOAPY_CONF)

# Set the scientific camera field-of-view to
# match the desired pixel scale
config.scis[0].pxlScale = pixelScale
config.calcParams()
config.tel.telDiam = diameter
config.calcParams()

print("[PARAMETER] Mirror diameter:", diameter, "m")
print("[PARAMETER] Pixel scale:", config.scis[0].pxlScale, "\'\'/px")
print("[PARAMETER] Wavelength:", config.scis[0].wavelength, "m")
print("[PARAMETER] FOV:", config.scis[0].FOV, "\'\'")
print("[PARAMETER] Pupil size:", config.sim.pupilSize, "px")

# Pupil mask
mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)

# Generate the Zernike modal basis
Zs = aotools.zernikeArray(nbOfZernike + 1, config.sim.pupilSize, norm='rms')

# -- Generate set of random coefficients --
# uniformely distributed from [-1, 1]
ZsCoeffs = np.random.random((nbOfPsfs, nbOfZernike))*2 - 1

# Normalize to a given total wfs rms
normInNmRms = 2000
ZsCoeffs = np.array([ZsCoeffs[k, :] / np.abs(ZsCoeffs[k,:]).sum() *
                   normInNmRms for k in range(nbOfPsfs)])

# Create the PSF image, 128x128 + 2x2 padding FTT over sampling
psfObj = SCI.PSF(config, nSci=0, mask=mask)

# PSFs with aberrations
psfsSoapy = np.zeros((nbOfPsfs, psfObj.detector.shape[0], psfObj.detector.shape[1]))
psfsSoapy_outOfFocus = np.zeros((nbOfPsfs, psfObj.detector.shape[0], psfObj.detector.shape[1]))

# Defocused phase
defocus = 0.4*normInNmRms*Zs[3,:,:]

# Reference
aberrations = np.zeros((128,128))
unaberrated = np.copy(psfObj.frame(aberrations))
hdu_primary = fits.PrimaryHDU(np.zeros(nbOfZernike))
unaberrated_outOfFocus = np.copy(psfObj.frame(aberrations+defocus))
hdu_In = fits.ImageHDU(unaberrated, name='INFOCUS')
hdu_Out = fits.ImageHDU(unaberrated_outOfFocus, name='OUTFOCUS')
hdu = fits.HDUList([hdu_primary, hdu_In, hdu_Out])
hdu.writeto('psf_unaberrated.fits', overwrite=True)

# Propagate aberrations
t0 = time.time()
for i in range(nbOfPsfs):
    # In focus
    aberrations = np.squeeze(np.sum(ZsCoeffs[i, :, None, None] * Zs[1:, :, :], axis=0))
    psfsSoapy[i, :, :] = np.copy(psfObj.frame(aberrations))
    # Out of focus
    abberations_outOfFocus = np.squeeze(aberrations)+defocus
    psfsSoapy_outOfFocus[i, :, :] = np.copy(psfObj.frame(abberations_outOfFocus))
    if i%(int(nbOfPsfs/10.)) == 0:
        print('[SOAPY] Propagation: ', i, '/', nbOfPsfs)
t_soapy = time.time() - t0
print('[SOAPY] Propagation: {0:2f}s'.format(t_soapy))

# Save PSFs
t0 = time.time()
for i in range(nbOfPsfs):
    outfile = "psfs/psf_" + str(i) + ".fits"
    hdu_primary = fits.PrimaryHDU(ZsCoeffs[i, :])
    hdu_In = fits.ImageHDU(psfsSoapy[i, :, :], name='INFOCUS')
    hdu_Out = fits.ImageHDU(psfsSoapy_outOfFocus[i, :, :], name='OUTFOCUS')
    hdu = fits.HDUList([hdu_primary, hdu_In, hdu_Out])
    hdu.writeto(outfile, overwrite=True)
t_soapy = time.time() - t0
print('[SOAPY] Saving: {0:2f}s'.format(t_soapy))



