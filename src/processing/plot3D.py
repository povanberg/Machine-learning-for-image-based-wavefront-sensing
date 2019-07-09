import matplotlib
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import scipy
from scipy import interpolate
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sph2cart(r, theta, phi):
    '''spherical to Cartesian transformation.'''
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def sphview(ax):
    '''returns the camera position for 3D axes in spherical coordinates'''
    r = np.square(np.max([ax.get_xlim(), ax.get_ylim()], 1)).sum()
    theta, phi = np.radians((90-ax.elev, ax.azim))
    return r, theta, phi

def getDistances(view,xpos, ypos, dz):
    distances  = []
    a = np.array((xpos, ypos, dz))
    for i in range(len(xpos)):
        distance = (a[0, i] - view[0])**2 + (a[1, i] - view[1])**2 + (a[2, i] - view[2])**2
        distances.append(np.sqrt(distance))
    return distances

def plot3D_continuous(im):

	font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 14}
	matplotlib.rc('font', **font)

	# Interpolate to smooth
	psf = scipy.ndimage.zoom(im, 3)

	plt.rcParams['grid.color'] = "lightgray"
	plt.rcParams['grid.linestyle'] = '--'
	fig = plt.figure(figsize=(10, 7))
	ax1 = fig.add_subplot(111, projection='3d')
	_x = np.arange(psf.shape[0])
	_y = np.arange(psf.shape[1])
	x, y = np.meshgrid(_x, _y)
	cmap = plt.cm.get_cmap('jet')
	#ax1.set_xlim3d(-64, 196)
	#x1.set_ylim3d(-64, 196)
	#ax1.set_zlim3d(0,1)
	ax1.set_xlabel("x")
	ax1.set_ylabel("y")
	ax1.w_xaxis.set_pane_color((1, 1, 1, 1.0))
	ax1.w_yaxis.set_pane_color((1, 1, 1, 1.0))
	ax1.w_zaxis.set_pane_color((1, 1, 1, 1.0))
	#ax1.set_xticks([])
	#ax1.set_yticks([])
	#ax1.set_zticks([])
	ax1.set_xticklabels([])
	ax1.set_yticklabels([])
	ax1.set_zticklabels([])
	ax1.w_zaxis.line.set_lw(0.)
	ax1.grid(True, color='lightgray', linestyle='--')
	st = np.int(psf.shape[0]/4)
	sp = psf.shape[0]-np.int(psf.shape[0]/4)
	surf = ax1.plot_surface(x[st:sp,st:sp], y[st:sp,st:sp], psf[st:sp,st:sp], rstride=1, cstride=1, cmap=cmap, linewidth=1, antialiased=False)
	plt.show()

def plot3D_discrete(im):

	font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 14}
	matplotlib.rc('font', **font)
	l = 20
	psf = im[64-l:64+l,64-l:64+l]

	fig = plt.figure(figsize=(10, 10))
	ax1 = fig.add_subplot(111, projection='3d')
	_x = np.arange(psf.shape[0])
	_y = np.arange(psf.shape[1])
	x, y = np.meshgrid(_x, _y)

	xpos = x.flatten()   # Convert positions to 1D array
	ypos = y.flatten()
	zpos = np.zeros(psf.shape[0]*psf.shape[1])

	dx = 1* np.ones_like(zpos)
	dy = dx.copy()
	dz = psf.flatten()

	# generate colors
	import matplotlib.colors as colors
	import matplotlib.cm as cmx

	cmap = plt.cm.jet # Get desired colormap - you can change this!
	max_height = np.max(dz)   # get range of colorbars so we can normalize
	min_height = np.min(dz)
	# scale each z to [0,1], and get their rgb values
	rgba = [cmap((k-min_height)/max_height) for k in dz]
	c_id = dz.argsort()

	# Get the camera's location in Cartesian coordinates.
	ax1.view_init(30, -115)
	x1, y1, z1 = sph2cart(*sphview(ax1))
	camera = np.array((x1,y1,0))
	# Calculate the distance of each bar from the camera.
	z_order = getDistances(camera, xpos, ypos, dz)
	max = np.max(z_order)

	n = psf.shape[0]*psf.shape[1]
	for i in range(n):
		pl = ax1.bar3d(xpos[i], ypos[i], zpos[i], dx[i], dy[i], dz[i],
		         color=rgba[i], alpha=1, zsort='max')
		# The z-order must be set explicitly.
		#
		# z-order values are somewhat backwards in magnitude, in that the largest
		# value is closest to the camera - unlike, in say, a coordinate system.
		# Therefore, subtracting the maximum distance from the calculated distance
		# inverts the z-order to the proper form.
		pl._sort_zpos = max - z_order[i]

	#ax1.bar3d(xpos,ypos,zpos, dx, dy, dz, color='0.85', zsort='max')
	plt.rcParams['grid.color'] = "lightgray"
	plt.rcParams['grid.linestyle'] = '--'
	ax1.set_xlabel("x")
	ax1.set_ylabel("y")
	ax1.w_xaxis.set_pane_color((1, 1, 1, 1.0))
	ax1.w_yaxis.set_pane_color((1, 1, 1, 1.0))
	ax1.w_zaxis.set_pane_color((1, 1, 1, 1.0))
	#ax1.set_xticks([])
	#ax1.set_yticks([])
	#ax1.set_zticks([])
	ax1.set_xticklabels([])
	ax1.set_yticklabels([])
	ax1.set_zticklabels([])
	#ax1.w_zaxis.line.set_lw(0.)
	plt.autoscale(enable=True, axis='both', tight=True)
	plt.show()

