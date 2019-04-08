import numpy as np
import numpy.fft as FFT
import aotools

def meterToRadian(array, wavelength):
    '''
        Convert array from meter to radian

    :param array: [nm]
    :param wavelength: [nm]
    :return: [rad]
    '''
    array_rad = (array / wavelength) * (2*np.pi)
    return array_rad

def getPhase(z_coeffs, z_basis):
    '''
        Compute phase from Zernike basis and Zernike coeffs

    :param z_coeffs: [rad]
    :param z_basis: [rad]
    :return: [rad]
    '''
    phase = z_coeffs[:, None, None] * z_basis[:, :, :]
    phase = np.sum(phase, axis=0)
    phase = np.squeeze(phase)
    return phase

def fft(array):
    '''
        Compute discrete fast fourier transform

    :param array:
    :return:
    '''
    fft_array = FFT.fftshift(FFT.fft2(FFT.fftshift(array)))
    return fft_array

def ifft(array):
    '''
        Compute discrete inverse fast fourier transform

    :param array:
    :return:
    '''
    fft_array = FFT.ifftshift(FFT.ifft2(FFT.ifftshift(array)))
    return fft_array

def pad_with(vector, pad_width, iaxis, kwargs):
    '''
        Padding utils

    :param vector:
    :param pad_width:
    :param iaxis:
    :param kwargs:
    :return:
    '''
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def addPadding(array, padding=2):
    '''
        Add padding to array

    :param array:
    :param padding:
    :return:
    '''
    size = array.shape[1]
    padded_array = np.pad(array, padding*size, pad_with, padder = 0)
    return padded_array

def removePadding(array, padding=2):
    '''
        Remove padding from array

    :param array:
    :param padding:
    :return:
    '''
    size = array.shape[1] // (2*padding + 1)
    rmPixel = padding*size
    return array[rmPixel:size+rmPixel,rmPixel:size+rmPixel]


def rootMeanSquaredError(array1, array2, mask=True):
    '''
        RMSE error between array1 and array2
        if mask=True computed over circle

    :param array:
    :return:
    '''
    if mask is True:
        size = array1.shape[1]
        center = size//2
        radius = size//2

        n = 0
        error = 0.0
        for x in range(size):
            for y in range(size):
                if (x-center)**2 + (y-center)**2 <= radius:
                    n += 1
                    error += (array1[x, y]-array2[x, y])**2
        rms_error = np.sqrt((1/n)*(error))
    else:
        rms_error = np.sqrt(((array1 - array2) ** 2).mean())
    return rms_error

def strehl(phase):
    mask = aotools.circle(64, 128)
    N= 0.0
    phase_mean = 0.0
    for i in range(128):
         for j in range(128):
             if(mask[i,j]>= 0.001):
                 N += 1
                 phase_mean += phase[i, j]
    phase_mean = phase_mean / N
    strehl = np.abs(np.mean(np.exp(1j*(phase-phase_mean))))**2
    return strehl