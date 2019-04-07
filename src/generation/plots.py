import time
import aotools
from radial import radial_data
import numpy as np
from scipy import fftpack
from astropy.io import fits
from soapy import SCI, confParse
from matplotlib import pyplot as plt

id = 0
phase = np.squeeze(np.sum(c_zernike[id, :, None, None] * zernike_basis[1:, :, :], axis=0))
F1 = fftpack.fft2(phase)
F2 = fftpack.fftshift( F1 )
psd2D = np.abs( F2 )**2

plt.imshow(np.sqrt(psfs_in[id,:,:]), cmap=plt.cm.jet)
plt.axis('off')
plt.savefig('psf_in.pdf')
plt.imshow(np.sqrt(psfs_out[id,:,:]), cmap=plt.cm.jet)
plt.axis('off')
plt.savefig('psf_out.pdf')
plt.imshow(phase, cmap=plt.cm.jet)
plt.axis('off')
plt.savefig('phase_in.pdf')

fig, ax = plt.subplots(figsize=(15, 5))
width = 0.4
plt.bar(i_zernike[:100], np.abs(c_zernike[id]/2200*2*np.pi)[:100], color='#32526e', width=width, zorder=3)
#plt.title('Zernike coefficient distribution', fontsize=19)
plt.xlabel('zernike coefficients', fontsize=16)
plt.ylabel('magnitude [rad]', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(zorder=0, color='lightgray', linestyle='--')
plt.ylim(0,0.4)
plt.savefig('z_distrib.pdf')
plt.show()

rad_obj = radial_data(psd2D, rmax=64)
fig, ax = plt.subplots()
plt.xlabel('Spatial frequency (cycles/pupil)', fontsize=13)
plt.ylabel('PSF (nm²nm²)', fontsize=13)
plt.loglog(rad_obj.r[1:], psd2D[65:128, 64])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(zorder=0, color='lightgray', linestyle='--')
start, end = ax.get_xlim()
plt.xticks(np.logspace(np.log10(start), np.log10(end), num=9, base=10),('10⁰','','','','10¹','','','','',''))
plt.savefig('PSD_rad.pdf')
plt.show()

fig, ax = plt.subplots()
#plt.title('1D PSD avg', fontsize=15)
plt.xlabel('Spatial frequency (cycles/pupil)', fontsize=13)
plt.ylabel('PSF (nm²nm²)', fontsize=13)
plt.loglog(rad_obj.r[1:],rad_obj.mean[1:])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(zorder=0, color='lightgray', linestyle='--')
start, end = ax.get_xlim()
plt.xticks(np.logspace(np.log10(start), np.log10(end), num=9, base=10),('10⁰','','','','10¹','','','','',''))
plt.savefig('PSD_rad_avg.pdf')
plt.show()
