import aotools
from astropy.io import fits
import numpy as np
import utils
from animation import *
from time import time

def GerchbergSaxton(target, source, phase, n_max=200, animation=True):
    '''
        Phase retrieval, Gerchberg-Saxton algorithm.

        [1] R. W. Gerchberg and W. O. Saxton, “A practical algorithm
            for the determination of the phase from image and diffraction
            plane pictures,” Optik 35, 237 (1972)

        [2] J. R. Fienup, "Phase retrieval algorithms: a comparison,"
            Appl. Opt. 21, 2758-2769 (1982)

    :param target:
    :param source:
    :param phase: Algorithm goal, provided for visualization and metrics
    :param n_max: Maximum number of iteration
    :param animation:
    :return:
    '''

    # Add padding
    target = utils.addPadding(np.sqrt(target))
    source = utils.addPadding(source)

    # Metrics: tuple -> (time, error)
    metrics = []

    # Initialize animation
    if animation:
        f, axarr = initAnimation()

    # Timer
    timer = 0.0

    # Random initializer
    A = source * np.exp(1j * 0.0 * np.pi * (np.random.rand(source.shape[0], source.shape[1])*2-1))

    for n in range(n_max):
        t0 = time()
        B = np.absolute(source) * np.exp(1j * np.angle(A))
        C = utils.fft(B)
        D = np.absolute(target) * np.exp(1j * np.angle(C))
        A = utils.ifft(D)

        t1 = time()
        timer += t1-t0

        phaseEst = source * np.angle(A)
        #phaseEst = np.rot90(np.rot90(-phaseEst))
        error = utils.rootMeanSquaredError(phase, utils.removePadding(phaseEst), mask=True)
        #error = utils.rootMeanSquaredError(C, D, mask=True)

        metrics.append((timer, error))

        if animation:
            H = utils.addPadding(mask) * np.exp(1j * (phaseEst-utils.addPadding(phase)))
            h = utils.fft(H)
            psf = utils.removePadding(np.abs(h) ** 2)
            updateAnimation(f, axarr, metrics, phase, utils.removePadding(phaseEst), psf, timer)

    return metrics

if __name__ == '__main__':

    # Files
    reference_file = 'references.fits'
    psf_file = 'psf_1.fits'

    # Data
    wavelength = 2200 * (10**(-9))  #[m]
    n=20
    z_basis = aotools.zernikeArray(n+1, 128, norm='rms') #[rad]

    rv_HDU = fits.open(reference_file)
    mask = rv_HDU[0].data # [0-1] function defining entrance pupil
    psf_reference = rv_HDU[1].data # diffraction limited point spread function

    HDU = fits.open(psf_file)
    phase = utils.meterToRadian(HDU[1].data, wavelength* (10**(9)))

    H = utils.addPadding(mask) * np.exp(1j * utils.addPadding(phase))
    h = utils.fft(H)
    psf_test = utils.removePadding(np.abs(h)**2)

    metrics = GerchbergSaxton(psf_test, mask, phase, n_max=200, animation=True)
    metrics = np.array(metrics)
