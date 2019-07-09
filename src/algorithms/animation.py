import aotools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils


def initAnimation():
    mpl.style.use('default')
    f, axarr = plt.subplots(2, 2, figsize=(10, 10))
    return f, axarr

def updateAnimation(f, axarr, error, phase, phaseEst, psf, timer):

    f.suptitle('Algorithm time: {0:.5}s'.format(timer))
    cmap = plt.cm.jet
    error = np.array(error)
    im1 = axarr[0, 0].plot(error[:, 1], linewidth=2.5)
    axarr[0, 0].grid(color='lightgrey', linestyle='--')
    axarr[0, 0].set_title("Wavefront error")
    axarr[0, 0].set_xlabel('iterations')
    axarr[0, 0].set_ylabel('RMSE')
    im2 = axarr[0, 1].imshow(psf**(1/3), cmap=cmap)
    cb2 = plt.colorbar(im2, ax=axarr[0, 1], fraction=0.046)
    axarr[0, 1].set_title("Point Spread function (strehl={0:.5f})".format(utils.strehl(phase-phaseEst)))
    axarr[0, 1].set_axis_off()
    mask=aotools.circle(64, 128).astype(np.float64)
    phase[mask<0.1]=None
    phaseEst[mask<0.1]=None
    im3 = axarr[1, 0].imshow(phase, cmap=cmap)
    im3.set_clim(-np.pi,np.pi)
    cb3 = plt.colorbar(im3, ax=axarr[1, 0], fraction=0.046)
    axarr[1, 0].set_title("Exact Phase")
    axarr[1, 0].set_axis_off()
    im4 = axarr[1, 1].imshow(phaseEst, cmap=cmap)
    im4.set_clim(-np.pi, np.pi)
    axarr[1, 1].set_title("Recovered phase")
    axarr[1, 1].set_axis_off()
    cb4 = plt.colorbar(im4, ax=axarr[1, 1], fraction=0.046)
    plt.pause(1e-5)
    axarr[0, 0].cla()
    cb2.remove()
    cb3.remove()
    cb4.remove()
    phase[mask<0.1]=0
    phaseEst[mask<0.1]=0
