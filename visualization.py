import sys
import time
import torch
import aotools
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from astropy.visualization import SqrtStretch, MinMaxInterval
from soapy import SCI, confParse

def visualize(dataset, model=None):

    wavelength = 2200 * (10**-9)
    
    # Soapy config
    SOAPY_CONF = "../../generation/psf.yaml"
    # Pixel size of science camera
    gridsize = 128
    # Telescope diameter
    diameter = 10
    # Pixel scale
    pixelScale = 0.01  # [''/px]s
    # Update scientific camera parameters
    config = confParse.loadSoapyConfig(SOAPY_CONF)
    config.scis[0].pxlScale = pixelScale
    config.tel.telDiam = diameter
    config.calcParams()
    # Circular pupil mask
    mask = aotools.circle(config.sim.pupilSize/2., config.sim.simSize)
    # Create the PSF image, 128x128 + 2x2 padding FTT over sampling
    psfObj = SCI.PSF(config, nSci=0, mask=mask)
    # norm
    norm = np.amax(psfObj.bestPSF)
    
    # Generate the Zernike modal basis
    zernike_basis = aotools.zernikeArray(20 + 1, 128, norm='rms')

    # Defocus 1/4 wavelength nm
    defocus = (wavelength/4) * (10**9) * zernike_basis[3, :, :]

    n = 3
    np.random.seed(44)
    n_id = [13, 6, 7]
    z_coeffs = np.zeros(shape=(n,20), dtype=np.float64)
    psf_in = np.zeros(shape=(n,128,128), dtype=np.float64)
    psf_out = np.zeros(shape=(n,128,128), dtype=np.float64)
    phase_in = np.zeros(shape=(n,128,128), dtype=np.float64)
    phase_out = np.zeros(shape=(n,128,128), dtype=np.float64)
    z_estimated = np.zeros(shape=(n,20), dtype=np.float64)
    p_estimated = np.zeros(shape=(n,128,128), dtype=np.float64)
    for i in range(n):
        z_coeffs[i,:] = dataset[n_id[i]]['zernike'].numpy()
        psf_in[i,:,:] = dataset[n_id[i]]['image'][0].numpy()
        psf_out[i,:,:] = dataset[n_id[i]]['image'][1].numpy() 
        phase_in[i,:,:] = np.squeeze(np.sum(z_coeffs[i, :, None, None] * zernike_basis[1:, :, :], axis=0))
        phase_out[i,:,:] = psf_in[i,:,:] + defocus  
        if model is not None:
            input_ = dataset[n_id[i]]['image'].unsqueeze(0)
            p_est, z_est = model(input_)
            p_estimated[i] = np.squeeze(p_est.detach().numpy())
            z_estimated[i] = np.squeeze(z_est.detach().numpy())
            
    stretch = MinMaxInterval() #+ SqrtStretch()
       
    print('Dataset Sample: Illustration with 3 random point spread functions')    
        
    print('Input:')    
        
    # In focus
    f, axarr = plt.subplots(1, n, figsize=(20, 5))
    for i in range(n):
        axarr[i].imshow(psf_in[i], cmap=plt.cm.jet)
        psf = np.copy(psfObj.frame(phase_in[i]))
        axarr[i].set_title("Strehl ratio: %f" % (strehl(phase_in[i])))
    plt.suptitle('Intensity In focus')
    plt.show()  
    
    # Out of Focus
    f, axarr = plt.subplots(1, n, figsize=(20, 5))
    for i in range(n):
        axarr[i].imshow(psf_out[i], cmap=plt.cm.jet)
        psf = np.copy(psfObj.frame(phase_out[i]))
        axarr[i].set_title("Strehl ratio: %f" % (strehl(phase_out[i])))
    plt.suptitle('Intensity Out of focus')
    plt.show()  

    # In phase 
    #f, axarr = plt.subplots(1, n, figsize=(20, 5))
    #for i in range(n):
    #    axarr[i].imshow(phase_in[i], cmap=plt.cm.jet)
    #    axarr[i].set_title('Corresponding phase: %f nm (rms error)' % (rms_wfe(phase_in[i])))
    #plt.show()  
   
    print('Output:')

    # Zernike coefficients distribution
    i_zernike = np.arange(2, 22)
    f, axarr = plt.subplots(1, n, figsize=(20, 3))
    w = 0.4
    for i in range(n):
        axarr[i].bar(i_zernike, np.abs(z_coeffs[i])/(wavelength*(10**9)), width=w, label='exact')
        if model is not None:
            axarr[i].bar(i_zernike + w, np.abs(z_estimated[i])/(wavelength*(10**9)), width=w, color='red', label='estimated')
        axarr[i].set_title("Zernike coefficients")
        axarr[i].set_ylabel('λ')
        axarr[i].legend()
    plt.show() 
    
    #Phase error
    print('Phase error: ideal - estimated. The values are expressed in nm (wavelength = 2200nm).')
    phase_error = phase_in-p_estimated
    f, axarr = plt.subplots(1, n, figsize=(20, 5))
    for i in range(n):
        im = axarr[i].imshow(np.abs(phase_error[i]), cmap=plt.cm.jet)
        f.colorbar(im, ax=axarr[i])
        axarr[i].set_title("Rms residual: %f nm / %f λ" % (rms_wfe(phase_error[i]),rms_wfe(phase_error[i])/2200))
    plt.show()  
    
    print('Reconstructed Point Spread function after phase correction')
    np.seterr(all='ignore')
    f, axarr = plt.subplots(1, n, figsize=(20, 5))
    for i in range(n):
        err = 0.01+np.add(phase_in[i],-p_estimated[i])
        psf = np.copy(psfObj.frame(err))
        im = axarr[i].imshow(np.sqrt(np.abs(psf)), cmap=plt.cm.jet)
        axarr[i].set_title("Strehl ratio: %f" % (strehl(phase_error[i])))
    plt.show()  
    
    print('Metrics: \n')
    
    exec_time = 0.0
    
    n = len(dataset)
    z_coeffs = np.zeros(shape=(n,20), dtype=np.float64)
    psf_in = np.zeros(shape=(n,128,128), dtype=np.float64)
    psf_out = np.zeros(shape=(n,128,128), dtype=np.float64)
    phase_in = np.zeros(shape=(n,128,128), dtype=np.float64)
    phase_out = np.zeros(shape=(n,128,128), dtype=np.float64)
    z_estimated = np.zeros(shape=(n,20), dtype=np.float64)
    p_estimated = np.zeros(shape=(n,128,128), dtype=np.float64)
    for i in range(n):
        z_coeffs[i,:] = dataset[i]['zernike'].numpy()
        psf_in[i,:,:] = dataset[i]['image'][0].numpy()
        psf_out[i,:,:] = dataset[i]['image'][1].numpy() 
        phase_in[i,:,:] = np.squeeze(np.sum(z_coeffs[i, :, None, None] * zernike_basis[1:, :, :], axis=0))
        phase_out[i,:,:] = psf_in[i,:,:] + defocus  
        if model is not None:
            input_ = dataset[i]['image'].unsqueeze(0)
            time1 = time.time()
            p_est, z_est = model(input_)
            exec_time += (time.time() - time1) / n
            p_estimated[i] = np.squeeze(p_est.detach().numpy())
            z_estimated[i] = np.squeeze(z_est.detach().numpy())
            
            
    #Average
    rms_phase_in = 0.0
    rms_phase_out = 0.0
    strehl_in = 0.0
    strehl_out = 0.0
    rms_corr = 0.0
    rms_z = 0.0
    strehl_corr = 0.0
    for i in range(n):
        rms_phase_in += rms_wfe(phase_in[i]) / n
        rms_phase_out += rms_wfe(phase_out[i]) / n
        strehl_in += strehl(phase_in[i]) / n
        strehl_out += strehl(phase_out[i]) / n
        rms_corr += rms_wfe(phase_in[i]-p_estimated[i]) / n
        rms_z += np.sqrt(np.mean((z_coeffs[i]-z_estimated[i])**2)) / n
        strehl_corr += strehl(phase_in[i]-p_estimated[i]) / n
        
    print('Average in focus strehl: %f' % (strehl_in))  
    print('Average out focus strehl: %f' % (strehl_out))  
    print('Average in focus wfe rms: %f nm / %f λ' % (rms_phase_in, rms_phase_in/2200))
    print('Average out focus wfe rms: %f nm / %f λ' % (rms_phase_out, rms_phase_out/2200))
    print('\n After correction: \n')
    print('Average strehl: %f' % (strehl_corr))    
    print('Average zernike rms error: %f nm / %f λ' % (rms_z, rms_z/2200))
    print('Average phase wfe rms : %f nm / %f λ' % (rms_corr, rms_corr/2200))
    print('Average Inference time : %f s' % exec_time)
    
          
def rms_wfe(image):
    mask = aotools.circle(64, 128)
    N = 0
    rms_error = 0.0
    for i in range(128):
         for j in range(128):
             if(mask[i,j]>= 0.001):
                 N += 1
                 rms_error += image[i, j]**2
    rms_error = rms_error/ N
    return np.sqrt(rms_error)
                           
def strehl(phase):
    mask = aotools.circle(64, 128)
    phase_rad = 2*np.pi*phase/2200
    N= 0.0
    phase_mean = 0.0
    for i in range(128):
         for j in range(128):
             if(mask[i,j]>= 0.001):
                 N += 1
                 phase_mean += phase_rad[i, j]
    phase_mean = phase_mean / N
    strehl = np.abs(np.mean(np.exp(1j*(phase_rad-phase_mean))))**2
    return strehl
