import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from spectral_cube import SpectralCube
from astropy import units as u
from astropy.io import fits

# plotting parameters
params = {'font.family' : 'serif',
			 'font.size' : 12,
			 'errorbar.capsize' : 3,
			 'lines.linewidth'   : 1.0,
			 'xtick.top' : True,
			 'ytick.right' : True,
			 'legend.fancybox' : False,
			 'xtick.major.size' : 4.0 ,
          'xtick.minor.size' : 2.0,    
          'ytick.major.size' : 4.0 ,
          'ytick.minor.size' : 2.0, 
			 'xtick.direction' : 'out',
			 'ytick.direction' : 'out',
			 'xtick.color' : 'black',
			 'ytick.color' : 'black',
			 'mathtext.rm' : 'serif',
          'mathtext.default': 'regular', 
			}
plt.rcParams.update(params)

# constants
c = 299792.458 #km/s

def load_parameters(datafile):
	# load observation parameters from fits header:
	# systemic velocity (vlsr)
	# rest frequency (restfreq)
	# beam size (bmin, bmaj)
	
	hdu = fits.open(datafile)[0]
	vlsr = (hdu.header['VELO-LSR'] / 1000.0) #km/s
	restfreq = hdu.header['RESTFREQ'] / (10.0**9) #GHz
	bmin = hdu.header['BMIN'] * 3600.0 #arcsec
	bmaj = hdu.header['BMAJ']* 3600.0 #arcsec
	return vlsr, restfreq, bmin, bmaj
	
def extract_average_spectrum(datafile,restfreq,vlsr,min_DEC_pixel,max_DEC_pixel,min_RA_pixel,max_RA_pixel):
	# extract average spectrum from data cube (X and Y)
	# frequency unit (X): GHz (corrected for systemic velocity vlsr)
	# flux unit (Y): Jy/beam
	
	cube = SpectralCube.read(datafile)  
	cube2 = cube.with_spectral_unit(u.GHz, velocity_convention='radio',rest_value=restfreq * u.GHz)
	subcube = cube2[:, min_DEC_pixel:max_DEC_pixel+1, min_RA_pixel:max_RA_pixel+1] 
	X=subcube.spectral_axis.value/(1.0 - (vlsr/c)) 
	Y = subcube.mean(axis=(1,2)).value  
	np.savetxt('spectrum.dat', np.c_[X,Y])
	return X, Y
	
def determine_noise(X, Y, min_channel, max_channel,show_plot=True):
	# compute noise in extracted average average spectrum (sigma)
	# noise should be determined in a line-free channel range (can be inspected with show_plot=True option)
	
	X_noise = X[min_channel:max_channel+1]
	X_noise_low = np.round(min(X_noise),decimals=3)
	X_noise_upp = np.round(max(X_noise),decimals=3)
	Y_noise = Y[min_channel:max_channel+1]
	sigma = np.round(np.std(Y_noise), decimals=5) 
	print('Noise in average spectrum: ' + str(sigma) + ' Jy/beam')
	print('determined between ' + str(X_noise_low) + ' and ' + str(X_noise_upp) + ' GHz')
	if show_plot == True: 
		plt.figure()
		ax = plt.subplot(111)
		plt.plot(X_noise,Y_noise,'black')
		ax.axhline(y=0, xmin=0, xmax=1, ls='-', color='red',lw=0.5)
		plt.xlabel('Frequency [GHz]')
		plt.ylabel('Flux Density [Jy/beam]')   
		plt.show()
		plt.close()
	return sigma
	
def load_line_table(data_tab):
	# load table with identified transitions 
	# first column: rest frequency (GHz)
	# blended transitions are labeled with 'blended'
	# second column: transition label (LATEX Format)
	
	freq = data_tab[:,0]
	molec = data_tab[:,1]
	frequency = np.zeros(freq.size)
	for i in range(0,freq.size):
		if freq[i] != 'blended':
			frequency[i] = float(freq[i])
	return molec, frequency
	
def annotate_lines(frequency,X,Y,sigma,molec,Y_low_lim,Y_upp_lim):
	# annotate transitions in plot
	# annotate strong lines below spectrum & weak lines above spectrum (set by threshold parameter)
	# y values at xytext in plt.annotate() should be adjusted by hand in order to avoid overlapping of the labels
	threshold = 0.045*max(Y)
	
	line_flux = np.zeros(frequency.size)
	counter = 0
	counter2 = 0
	for i in range(0, frequency.size):
		if frequency[i] != 0:
			line_flux[i] = np.interp(frequency[i], X, Y)
			if np.interp(frequency[i], X, Y) >= threshold:
				counter = counter + 1
				if counter % 2 == 0:
					plt.annotate(molec[i], xy=(frequency[i], 0.0), xytext=(frequency[i], 0.15*Y_low_lim), arrowprops=dict(arrowstyle='-', relpos=(0,0),alpha=0.3, color='blue',linewidth=0.5), fontsize=7, rotation = 90,alpha=1.0, color='blue')
				if counter % 2 == 1:
					plt.annotate(molec[i], xy=(frequency[i], 0.0), xytext=(frequency[i], 0.6*Y_low_lim), arrowprops=dict(arrowstyle='-', relpos=(0,0),alpha=0.3, color='blue',linewidth=0.5), fontsize=7, rotation = 90,alpha=1.0, color='blue')
			else:
				counter2 = counter2 + 1
				if counter2 % 3 == 0:
					plt.annotate(molec[i], xy=(frequency[i], line_flux[i] + 0.005), xytext=(frequency[i], 0.9*Y_upp_lim), arrowprops=dict(arrowstyle='-', relpos=(0,0),alpha=0.3, color='blue',linewidth=0.5), fontsize=7, rotation = 90,alpha=1.0, color='blue')
				if counter2 % 3 == 1:
					plt.annotate(molec[i], xy=(frequency[i], line_flux[i] + 0.005), xytext=(frequency[i], 0.7*Y_upp_lim), arrowprops=dict(arrowstyle='-', relpos=(0,0),alpha=0.3, color='blue',linewidth=0.5), fontsize=7, rotation = 90,alpha=1.0, color='blue')
				if counter2 % 3 == 2:
					plt.annotate(molec[i], xy=(frequency[i], line_flux[i] + 0.005), xytext=(frequency[i], 0.5*Y_upp_lim), arrowprops=dict(arrowstyle='-', relpos=(0,0),alpha=0.3, color='blue',linewidth=0.5), fontsize=7, rotation = 90,alpha=1.0, color='blue')

def plot_spectrum(X,Y,sigma,frequency,molec,restfreq,bmin,bmaj):
	Y_low_lim = -0.5*max(Y)
	Y_upp_lim = 1.2*max(Y)
	
	plt.figure(figsize=(10,6))
	ax = plt.subplot(111)
	# plot average spectrum
	plt.plot(X,Y,'black',label='Data',linewidth=0.5)
	# plot 5-sigma line
	ax.axhline(y=5.0*sigma, xmin=0, xmax=1, ls='-', color='red',lw=0.5, label='5$\sigma$')
	# add legend
	ax.legend(numpoints=1, ncol=1,loc='upper left', fontsize=7)
	# annotate transitions
	annotate_lines(frequency,X,Y,sigma,molec,Y_low_lim,Y_upp_lim)
	# figure properties
	plt.xlabel('Frequency [GHz]')
	plt.ylabel('Flux Density [Jy/beam]')    
	plt.xlim(float(min(X))-0.015, max(X)+0.015)
	plt.ylim(Y_low_lim, Y_upp_lim)
	yminor = MultipleLocator(0.05)
	ax.yaxis.set_minor_locator(yminor)
	xmajor = MultipleLocator(0.5)
	ax.xaxis.set_major_locator(xmajor)
	xminor = MultipleLocator(0.1)
	ax.xaxis.set_minor_locator(xminor)
	
	# brightness temperature on right y-axis
	ax2 = ax.twinx()
	Jy_to_K=1.222*10**6/(restfreq**2*bmin*bmaj)
	ax2.set_ylim(Y_low_lim*Jy_to_K, Y_upp_lim*Jy_to_K)
	ax2.set_ylabel('Brightness Temperature [K]')
	yminor2 = MultipleLocator(5)
	ax2.yaxis.set_minor_locator(yminor2)
	ymajor2 = MultipleLocator(10)
	ax2.yaxis.set_major_locator(ymajor2)
	
	plt.savefig('Average_spectrum.pdf', format='pdf', bbox_inches='tight')
	plt.close()
	
###----REQUIRED INPUT----###
# input datacube	
datafile = 'afgl2591_widex_all_contsub_robust0p1_clark-notmerged-pbcor.fits'

# input table with identified transitions
data_tab=np.loadtxt('TransitionProperties.dat',dtype='U',comments='#', delimiter=' ')

# region which sets rectangle for average spectrum
min_DEC_pixel = 500
max_DEC_pixel = 520
min_RA_pixel = 500
max_RA_pixel = 520


# line-free channel range
min_channel = 100
max_channel = 180
###----REQUIRED INPUT END----###

vlsr, restfreq, bmin, bmaj = load_parameters(datafile)
X, Y = extract_average_spectrum(datafile,restfreq,vlsr,min_DEC_pixel,max_DEC_pixel,min_RA_pixel,max_RA_pixel)
sigma = determine_noise(X, Y, min_channel, max_channel,show_plot=True)
molec, frequency = load_line_table(data_tab)
plot_spectrum(X,Y,sigma,frequency,molec,restfreq,bmin,bmaj)