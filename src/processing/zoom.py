import aotools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def plot_with_zoom(im, sub=[48,80], name="test", save=False):

	font = {'family' : 'normal',
		'weight' : 'normal',
		'size'   : 14}
	matplotlib.rc('font', **font)
	extent = (0, 127, 0, 127)

	f, ax = plt.subplots(1, 1)
	f.subplots_adjust(left=0.2, bottom=0.2)
	im_ax = ax.imshow(im, cmap=plt.cm.jet)
	plt.xticks([0,20, 40, 60, 80, 100, 120])
	plt.yticks([0, 40, 60, 80, 100, 120])

	axins = zoomed_inset_axes(ax, 2, loc="lower left", bbox_to_anchor=(20,95), borderpad=5) # zoom = 6
	axins.imshow(im, extent=extent, interpolation="nearest", origin="lower", cmap=plt.cm.jet)

	# sub region of the original image
	x1, x2, y1, y2 = sub[0], sub[1], sub[0], sub[1]
	axins.set_xlim(x1, x2)
	axins.set_ylim(y1, y2)

	plt.xticks(visible=False)
	plt.yticks(visible=False)

	# draw a bbox of the region of the inset axes in the parent axes and
	# connecting lines between the bbox and the inset axes area
	mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.5")

	#cbar = plt.colorbar(im,fraction=0.046, pad=0.0.1)
	add_colorbar(im_ax)
	if save:
		plt.savefig(name)
		plt.close()
	else:
		plt.show()

def plot_map(phase, name="test", save=False):

	mask = aotools.circle(64, 128).astype(np.float64)
	phase[mask<0.1] = None 
	im = plt.imshow(phase, cmap=plt.cm.jet)
	plt.clim(-np.pi, np.pi)
	for d in ["left", "top", "bottom", "right"]:
		plt.gca().spines[d].set_visible(False)
	cbar = plt.colorbar(im, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
	cbar.ax.set_yticklabels(['$- \pi$','$- \pi$/2', '0','$\pi/2$', '$\pi$'])
	plt.xticks([0,20, 40, 60, 80, 100, 120])
	if save:
		plt.savefig(name)
		plt.close()
	else:
		plt.show()

