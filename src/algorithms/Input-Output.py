import aotools
from astropy.io import fits
import numpy as np
import utils
from animation import *
from time import time

def HybridInputOutput(target, source, phase, n_max=200, animation=True):
    '''

     [1] E. Osherovich, Numerical methods for phase retrieval, 2012,
        https://arxiv.org/abs/1203.4756
    [2] J. R. Fienup, Phase retrieval algorithms: a comparison, 1982,
        https://www.osapublishing.org/ao/abstract.cfm?uri=ao-21-15-2758

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

    # Metrics: tuple -> (time, rmse)
    metrics = []

    # Initialize animation
    if animation:
        f, axarr = initAnimation()

    # Timer
    timer = 0.0

    # Random initializer
    g_k_prime = np.exp(1j * 0.0 * np.pi * (np.random.rand(source.shape[0], source.shape[1])*2-1))


    # Previous iteration
    g_k_previous = None

    for n in range(n_max):
        t0 = time()

        g_k = source * np.exp(1j * np.angle(g_k_prime))
        G_k= utils.fft(g_k)
        G_k_prime = np.absolute(target) * np.exp(1j * np.angle(G_k))
        g_k_prime = utils.ifft(G_k_prime)


        if g_k_previous is None:
            g_k_previous = g_k_prime
        else:
            g_k_previous = g_k

        indices = np.logical_or(np.logical_and(g_k < 0, source),  np.logical_not(source))

        g_k[indices] = g_k_previous[indices] - 0.9 * np.real(g_k_prime[indices])

        t1 = time()
        timer += t1-t0

        phaseEst = source * np.angle(g_k)
        #phaseEst = np.rot90(np.rot90(-phaseEst))
        error = utils.rootMeanSquaredError(phase, utils.removePadding(phaseEst), mask=True)
        #error = utils.rootMeanSquaredError(G_k, G_k_prime, mask=True)

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

    metrics = HybridInputOutput(psf_test, mask, phase, n_max=300, animation=True)
    metrics = np.array(metrics)
