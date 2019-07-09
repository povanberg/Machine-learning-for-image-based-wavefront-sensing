import time
import aotools
from radial import radial_data
import numpy as np
from scipy import fftpack
from astropy.io import fits
from soapy import SCI, confParse
from matplotlib import pyplot as plt

# ------------------------------------------------------------------------
# Generate Point Spread functions from randomly drawn non-common path
# aberrations. The aberration follows a 1/f^2 law.
# One PSF in focus as well as a PSF out of focus are saved in FITS format
# (see astropy docs). The corresponding phase and Zernike Coefficient
# are also saved.
# ------------------------------------------------------------------------

np.random.seed(seed=0)

SOAPY_CONF = "psf.yaml"                         # Soapy config
gridsize = 128                                  # Pixel size of science camera
wavelength = 2.2e-6                             # Observational wavelength
diameter = 10                                   # Telescope diameter
pixelScale = 0.01                               # [''/px]s

n_psfs = 5                                      # Number of PSFs
n_zernike = 100                                 # Number of Zernike polynomials
i_zernike = np.arange(2, n_zernike + 2)         # Zernike polynomial indices (piston excluded)
o_zernike= []                                   # Zernike polynomial radial Order, see J. Noll paper :
for i in range(1,n_zernike):                    # "Zernike polynomials and atmospheric turbulence", 1975
    for j in range(i+1):
        if len(o_zernike) < n_zernike:
            o_zernike.append(i)

# Generate randomly Zernike coefficient. By dividing the value
# by its radial order we produce a distribution following
# the expected 1/f^-2 law.
c_zernike = 2 * np.random.random((n_psfs, n_zernike)) - 1
for j in range(n_psfs):
    for i in range(n_zernike):
        c_zernike[j, i] = c_zernike[j, i] / o_zernike[i]
c_zernike = np.array([c_zernike[k, :] / np.abs(c_zernike[k, :]).sum()
                      * wavelength*(10**9) for k in range(n_psfs)])

# Update scientific camera parameters
config = confParse.loadSoapyConfig(SOAPY_CONF)
config.scis[0].pxlScale = pixelScale
config.tel.telDiam = diameter
config.calcParams()

mask = aotools.circle(config.sim.pupilSize / 2., config.sim.simSize).astype(np.float64)
zernike_basis = aotools.zernikeArray(n_zernike + 1, config.sim.pupilSize, norm='rms')

psfObj = SCI.PSF(config, nSci=0, mask=mask)

psfs_in = np.zeros((n_psfs, psfObj.detector.shape[0], psfObj.detector.shape[1]))
psfs_out = np.zeros((n_psfs, psfObj.detector.shape[0], psfObj.detector.shape[1]))

defocus = (wavelength / 4) * (10 ** 9) * zernike_basis[3, :, :]

t0 = time.time()
n_fail = 0

for i in range(n_psfs):

    aberrations_in = np.squeeze(np.sum(c_zernike[i, :, None, None] * zernike_basis[1:, :, :], axis=0))
    psfs_in[i, :, :] = np.copy(psfObj.frame(aberrations_in.astype(np.float64)))

    aberations_out = np.squeeze(aberrations_in) + defocus
    psfs_out[i, :, :] = np.copy(psfObj.frame(aberations_out.astype(np.float64)))

    # psfs_in[i, :, :] = np.random.poisson(lam=100000*psfs_in[i, :, :], size=None)
    # psfs_out[i, :, :] = np.random.poisson(lam=100000*psfs_out[i, :, :], size=None)

    # Save
    outfile = "psf_" + str(i) + ".fits"
    hdu_primary = fits.PrimaryHDU(c_zernike[i, :].astype(np.float32))
    hdu_phase = fits.ImageHDU(aberrations_in.astype(np.float32), name='PHASE')
    hdu_In = fits.ImageHDU(psfs_in[i, :, :].astype(np.float32), name='INFOCUS')
    hdu_Out = fits.ImageHDU(psfs_out[i, :, :].astype(np.float32), name='OUTFOCUS')
    hdu = fits.HDUList([hdu_primary, hdu_phase, hdu_In, hdu_Out])
    hdu.writeto(outfile, overwrite=True)

    t_soapy = time.time() - t0
print('Propagation and saving finished in {0:2f}s'.format(t_soapy))
print('Failed: {0:2f}'.format(n_fail))
