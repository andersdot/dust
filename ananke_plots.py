import matplotlib as mpl
mpl.use('Agg')

#specify the base directory holding the surveys here (with trailing slash please)
dirbase='/Collections/Ananke/ananke/'


#chunk size (avoids memory overflow)
dn = 10000000

#import some stuff used a lot
import numpy as np
import h5py
import glob
import sys

from matplotlib import pyplot as pl
pl.rc('text', usetex=True)
pl.rc('font', **{'family':'Computer Modern','size':20})
pl.rc('axes', labelsize=16)
pl.rc('xtick',labelsize=16)
pl.rc('ytick',labelsize=16)
from matplotlib.colors import LogNorm,PowerNorm
from matplotlib import cm


def histogram(key, bins, sim='m12f', nlsr=0, verbose=False):
	"""
	Makes a 1D histogram of a quantity in the user-specified survey, using user-specified bins. 
	Assumes that the file structure is the same as on yt hub.

	Arguments:
		key (string): name of field to make the histogram over
		bins (Nx1 array of floats): the edges of the bins to use
		sim (string): the name of the simulation used to make the synthetic survey. One of 'm12f', 'm12i', or 'm12m'
		nlsr (integer between 0 and 2 inclusive): the number corresponding to the local standard of rest used for the survey.
		verbose (boolean): optional argument, set to True to print progress

	Returns:
		H ((N-1)x1 array of floats): counts per bin for the desired quantity
		bins (Nx1 array of floats): bin edges supplied by user
	"""

	# create list of survey files to aggregate over
	dirname = dirbase+sim+'/lsr_'+str(nlsr)+'/'
	fname_pattern = dirname+'lsr-'+str(nlsr)+'-rslice-?.'+sim+'-res7100-md-sliced-gcat-dr2.hdf5'
	flist=glob.glob(fname_pattern)

	#check that there are actually files there
	if len(flist)<1:
		raise IOError('No files match '+ fname_pattern)

	#define aggregator histogram
	H = np.zeros_like(bins[1:])

	#loop over files
	for fn in flist:
		if verbose: print fn
		with h5py.File(fn,mode='r') as f:
			n = f['l'].len()
			#loop over chunks
			for chunk in range(0,n,dn):
				dc = min(chunk+dn,n)
				if verbose: print chunk, dc
				#add to histogram
				h, xe = np.histogram(f[key][chunk:dc],bins=bins)
				H+=h

	return H, bins


def histogram2d(key_x, key_y, xe, ye, sim, nlsr, verbose=False):
	"""
	Makes a 2D histogram of two quantities in the user-specified survey, using user-specified bins. 
	Assumes that the file structure is the same as on yt hub.

	Arguments:
		key_x, key_y (string): name of field for the x,y coordinate of the histogram
		xe, ye (Nx1, Mx1 array of floats): the x and y edges of the bins to use
		sim (string): the name of the simulation used to make the synthetic survey. One of 'm12f', 'm12i', or 'm12m'
		nlsr (integer between 0 and 2 inclusive): the number corresponding to the local standard of rest used for the survey.
		verbose (boolean): optional argument, set to True to print progress

	Returns:
		H ((M-1)x(N-1) array of floats): counts per bin for the desired quantity
		X, Y ((N-1)x(M-1) arrays of floats): meshgrid of center points of the histogram
	"""

	# create list of survey files to aggregate over
	dirname = dirbase+sim+'/lsr_'+str(nlsr)+'/'
	fname_pattern = dirname+'lsr-'+str(nlsr)+'-rslice-?.'+sim+'-res7100-md-sliced-gcat-dr2.hdf5'
	flist=glob.glob(fname_pattern)

	if len(flist)<1:
		raise IOError('No files match '+ fname_pattern)

	#define aggregator histogram
	X,Y = np.meshgrid(0.5*(xe[:-1]+xe[1:]),0.5*(ye[:-1]+ye[1:]))
	H = np.zeros_like(X.T)

	#loop over files
	for fn in flist:
		if verbose: print fn
		with h5py.File(fn,mode='r') as f:
			n = f['l'].len()
			#loop over chunks
			for chunk in range(0,n,dn):
				dc = min(chunk+dn,n)
				if verbose: print chunk, dc
				#add to histogram
				h, xe, ye = np.histogram2d(f[key_x][chunk:dc],f[key_y][chunk:dc],bins=[xe,ye])
				H+=h

	return H, X, Y


def galactic_map(sim, nlsr):
	"""
	Makes an all-sky star count map for one synthetic survey specified by the user.
	(One panel of Figure 6)
	Also saves histograms of fluxes for use making the RGB flux maps of Figure 7.

	Arguments:
		sim (string): the name of the simulation used to make the synthetic survey. One of 'm12f', 'm12i', or 'm12m'
		nlsr (string): the number corresponding to the local standard of rest used for the survey. One of '0', '1', '2'
	Returns:
		None
	Output:
		Saves individual .npz files with the count, density, and flux histograms computed for each slice, plus the figure.
	"""

	da = 0.04
	xe = np.arange(-180.,180.+da,da)
	ye = np.arange(-90.,90.+da,da)
	X,Y = np.meshgrid(xe[:-1],ye[:-1])

	H_counts_all = np.zeros_like(X.T)
	H_dens_all = np.zeros_like(X.T)
	H_G_all = np.zeros_like(X.T)
	H_GBp_all = np.zeros_like(X.T)
	H_GRp_all = np.zeros_like(X.T)

	dirname = dirbase+sim+'/lsr_'+nlsr+'/'
	fname_pattern = dirname+'lsr-'+nlsr+'-rslice-?.'+sim+'-res7100-md-sliced-gcat-dr2.hdf5'
	flist=glob.glob(fname_pattern)
	if len(flist)<1:
		raise IOError('No files match '+ fname_pattern)
	for fn in flist:
		print fn
		with h5py.File(fn,mode='r') as f:
			n = f['l'].shape[0]
			H_counts = np.zeros_like(X.T)
			H_G = np.zeros_like(X.T)
			H_GBp = np.zeros_like(X.T)
			H_GRp = np.zeros_like(X.T)
			for chunk in range(0,n,dn):
				dc = min(chunk+dn,n)
				print chunk, dc

				#do source count histogram
				h, xe, ye = np.histogram2d(f['l'][chunk:dc],f['b'][chunk:dc],bins=[xe,ye])
				H_counts+=h

				#now do flux-weighted

				h,xe,ye = np.histogram2d(f['l'][chunk:dc],f['b'][chunk:dc],
					weights=10**(-0.4*f['phot_bp_mean_mag'][chunk:dc]),bins=(xe,ye))
				H_GBp+=h

				h,xe,ye = np.histogram2d(f['l'][chunk:dc],f['b'][chunk:dc],
					weights=10**(-0.4*f['phot_rp_mean_mag'][chunk:dc]),bins=(xe,ye))
				H_GRp+=h

				h,xe,ye = np.histogram2d(f['l'][chunk:dc],f['b'][chunk:dc],
					weights=10**(-0.4*f['phot_g_mean_mag'][chunk:dc]),bins=(xe,ye))
				H_G+=h
		sys.stdout.flush()
		#convert to counts/squared arcmin
		H_dens=H_counts/(da*60.)**2
		rnum = fn.split('rslice-')[1][0]
		savename = dirname+'figures/galactic-map-'+sim+'-lsr-'+nlsr+'-rslice-'+rnum
		print 'Writing to', savename
		print 'min level', H_counts.min()
		print 'max level', H_counts.max()

		np.savez(savename,
	             X=X,
	             Y=Y,
	             H_counts=H_counts,
	             H_dens=H_dens,
	             H_G=H_G,
	             H_GRp=H_GRp,
	             H_GBp=H_GBp)
	    
		H_counts_all += H_counts
		H_dens_all += H_dens
		H_G_all += H_G
		H_GRp_all += H_GRp
		H_GBp_all += H_GBp

	fig = pl.figure(figsize=(7,4))

	ax = pl.subplot(111,projection='hammer')
	#need to convert coordinates to radians if using projection
	X_rad = np.radians(X)
	Y_rad = np.radians(Y)
	cbh = ax.pcolormesh(X_rad,Y_rad,H_dens_all.T,norm=PowerNorm(gamma=0.5),cmap='gist_gray',vmin=0,vmax=H_counts.max())
	ax.grid('off')
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
	pl.colorbar(cbh,label='sources/squared arcmin',orientation='vertical',fraction=0.03,aspect=60)

	fig.tight_layout()

	savename = dirname+'figures/galactic-map-'+sim+'-lsr-'+nlsr
	fig.savefig(savename+'.png',dpi=300)


def toomre_diagram(sim, nlsr):
	"""
	Makes a two-panel figure showing the Toomre diagram of a synthetic survey (a la Figure 9).

	Arguments:
		sim (string): the name of the simulation used to make the synthetic survey. One of 'm12f', 'm12i', or 'm12m'
		nlsr (string): the number corresponding to the local standard of rest used for the survey. One of '0', '1', '2'
	Returns:
		None
	Output:
		Saves the arrays with the information to make the plots as .npz files, plus the figure.
	"""

	import astropy.coordinates as apc
	from astropy import units as u
	from astropy.coordinates import ICRS, Galactocentric, CartesianDifferential

	#LSR locations of surveys (needed to compute galactocentric velocities)
	lsr_xv = {
	'm12m':
		[[0.000000, 8.200000, 0.000000, 254.918686, 16.790098, 1.964817],
		 [-7.101408, -4.100000, 0.000000, -128.247955, 221.148926, 5.850575],
		 [7.101408, -4.100000, 0.000000, -106.620308, -232.205551, -6.418519]
		],
	'm12f':
		[[0.000000, 8.200000, 0.000000, 226.184921, 14.377288, -4.890565],
		 [-7.101408, -4.100000, 0.000000, -114.035072, 208.726669, 5.063526],
		 [7.101408, -4.100000, 0.000000, -118.143044, -187.763062, -3.890517]
		],
	'm12i':
		[[0.000000, 8.200000, 0.000000, 224.709198, -20.380102, 3.895417],
		 [-7.101408, -4.100000, 0.000000, -80.426880, 191.723969, 1.503948],
		 [7.101408, -4.100000, 0.000000, -87.273514, -186.856659, -9.460751]
		]
	}

	def get_vcen(sim,nlsr):
		xvsun = lsr_xv[sim][int(nlsr)]
		phi = np.pi + np.arctan2(xvsun[1], xvsun[0])
	  	rot = np.array([
	      [np.cos(phi), np.sin(phi), 0.0],
	      [-np.sin(phi), np.cos(phi), 0.0],
	      [0.0, 0.0, 1.0]
	      ])
	  	vctr_rot = np.dot(rot,xvsun[3:])

	  	return vctr_rot

	dn = 10000000
	dv = 1.0
	xe = np.arange(-500.0,500.0+dv,dv)
	ye = np.arange(0.0,500.0+dv,dv)
	x_cen, y_cen = 0.5*(xe[:-1]+xe[1:]),  0.5*(ye[:-1]+ye[1:])
	X,Y=np.meshgrid(x_cen,y_cen)

	H_toomre = np.zeros_like(X.T)
	H_feh  = np.zeros_like(X.T)

	dirname = dirname = dirbase+sim+'/lsr_'+nlsr+'/'
	fname_pattern = dirname+'lsr-'+nlsr+'-rslice-?.'+sim+'-res7100-md-sliced-gcat-dr2.hdf5'
	flist=glob.glob(fname_pattern)
	if len(flist)<1:
		raise IOError('No files match '+ fname_pattern)

	vctr_rot = get_vcen(sim,nlsr)
	gcen = Galactocentric(galcen_distance=8.2*u.kpc,
						z_sun=0.0*u.kpc,
						galcen_v_sun=CartesianDifferential(vctr_rot*u.km/u.s))
	for fn in flist:
		print fn
		with h5py.File(fn,mode='r') as f:
			n = f['l'].len()
			for chunk in range(0,n,dn):
				dc = min(chunk+dn,n)
				print chunk, dc

				hasrv = (np.isfinite(f['radial_velocity'][chunk:dc]))
				has_good_par = (f['parallax_over_error'][chunk:dc]>10)

				good=has_good_par&hasrv

				if good.sum() > 0:
					print 'making histograms with', good.sum(), 'sources'
					c = ICRS(dec=f['dec'][chunk:dc][good]*u.degree,
					ra=f['ra'][chunk:dc][good]*u.degree,
					distance=1.0/(f['parallax'][chunk:dc][good])*u.kpc,
					pm_dec = f['pmdec'][chunk:dc][good]*u.mas/u.yr,
					pm_ra_cosdec=f['pmra'][chunk:dc][good]*u.mas/u.yr,
					radial_velocity=f['radial_velocity'][chunk:dc][good]*u.km/u.s)

					gc = c.transform_to(gcen)
					U = gc.cartesian.differentials['s'].d_x.value
					V = gc.cartesian.differentials['s'].d_y.value
					W = gc.cartesian.differentials['s'].d_z.value

					h_c, xe, ye = np.histogram2d(V,np.sqrt((U)**2 + (W)**2),bins=[xe,ye])
					h_w, xe, ye = np.histogram2d(V,np.sqrt((U)**2 + (W)**2),bins=[xe,ye],
						weights=f['feh'][chunk:dc][good])

					H_toomre += h_c
					H_feh += h_w

	#compute weighted average for feh
	H_feh/=H_toomre
	H_feh[H_toomre<=0]=-99

	#compute density for toomre diagram
	H_toomre/=dv**2

	#save data
	savename = dirname+'figures/toomre-'+sim+'-lsr-'+nlsr
	print 'Writing to', savename
	print 'min level', H_toomre.min()
	print 'max level', H_toomre.max()

	np.savez(savename,
		xe=xe,
		ye=ye,
		X=X,
		Y=Y,
		x_cen=x_cen,
		y_cen=y_cen,
		H_feh=H_feh,
		H_toomre=H_toomre
		)

	fig,axs=pl.subplots(1,2,figsize=(8,5),sharex=True,sharey=True)

	im0 = axs[0].pcolormesh(X,Y,H_toomre.T,norm=LogNorm())
	axs[0].set_xlabel(r'$V_Y$ [km s${}^{-1}$]')
	axs[0].set_ylabel(r'$\sqrt{V_X^2+V_Z^2}$ [km s${}^{-1}$]')
	fig.colorbar(im0,ax=axs[0],label=r'counts (km/s)${}^{-2}$',orientation='horizontal')

	im1 = axs[1].pcolormesh(X,Y,H_feh.T,vmin=-2,vmax=0.3,cmap='inferno')
	axs[1].set_xlabel(r'$V_Y$ [km s${}^{-1}$]')
	fig.colorbar(im1,ax=axs[1],label='mean [Fe/H]',orientation='horizontal')

	fig.tight_layout()
	fig.savefig(savename+'.png', dpi=300)


def threecolor_map(sim,nlsr):
	"""
	Makes an RGB flux map of one of the synthetic surveys (a la Figure 7).
	Uses the data files written by galactic_map, so run that first.

	Arguments:
		sim (string): the name of the simulation used to make the synthetic survey. One of 'm12f', 'm12i', or 'm12m'
		nlsr (string): the number corresponding to the local standard of rest used for the survey. One of '0', '1', '2'
	Returns:
		None
	Output:
		Saves two versions of the image - with and without median filtering.
	"""

	from scipy.ndimage.filters import median_filter

	# Gaia DR2 zero points for revised passbands (Vega magnitudes)
	# Reference: Evans et al. 2018 
	# Used to compute fluxes for RGB maps
	g_zp, gbp_zp, grp_zp = (25.6914396869, 25.3488107670, 24.7626744847)
	gflux_zp = 10**(0.4*g_zp)
	gbpflux_zp = 10**(0.4*gbp_zp)
	grpflux_zp = 10**(0.4*grp_zp)

	fstem = dirbase+sim+'/lsr_'+nlsr+'/figures/galactic-map-'+sim+'-lsr-'+nlsr

	print 'loading from '+fstem
	data = [np.load(fstem+'-rslice-'+str(i)+'.npz') for i in range(0,10)]

	H_counts_all = np.array([data[i]['H_counts'] for i in range(0,10)]).sum(axis=0)

	H_G_all = np.array([data[i]['H_G'] for i in range(0,10)]).sum(axis=0)*gflux_zp
	H_GRp_all = np.array([data[i]['H_GRp'] for i in range(0,10)]).sum(axis=0)*grpflux_zp
	H_GBp_all = np.array([data[i]['H_GBp'] for i in range(0,10)]).sum(axis=0)*gbpflux_zp


	norm=np.array([np.percentile(H_GRp_all.flatten(),99.),np.percentile(H_G_all.flatten(),99.),np.percentile(H_GBp_all.flatten(),99.)])
	print norm

	im=np.transpose(np.array([H_GRp_all/norm[0],H_G_all/norm[1],H_GBp_all/norm[2]]),axes=(2,1,0))
	im[im>1]=1

	fig, ax = pl.subplots(1,1,figsize=(8,4))
	ax.imshow(im,cmap='RdBu',extent=(-180,180,-90,90))
	ax.text(0.95,0.95,
		sim+'-lsr'+nlsr, 
		ha='right', va='top',
		transform=ax.transAxes,
		fontsize=20, family='Helvetica',color='w',weight='bold')
	fig.tight_layout()
	fig.savefig(fstem+'-threecolor.png',dpi=300)

	im_filt = median_filter(im, size=2)

	fig, ax = pl.subplots(1,1,figsize=(8,4))
	ax.imshow(im_filt,cmap='RdBu',extent=(-180,180,-90,90))
	ax.text(0.95,0.95,
		sim+'-lsr'+nlsr, 
		ha='right', va='top',
		transform=ax.transAxes,
		fontsize=20, family='Helvetica',color='w',weight='bold')
	fig.tight_layout()
	fig.savefig(fstem+'-threecolor-medfilt.png',dpi=300)


def galactic_hrd_vtan(sim, nlsr):
	"""
	Makes an observational H-R diagram sliced by V_tan for one synthetic survey
	(a la Figure 8 and Gaia Collab, Babusiaux et al. 2018)

	Arguments:
		sim (string): the name of the simulation used to make the synthetic survey. One of 'm12f', 'm12i', or 'm12m'
		nlsr (string): the number corresponding to the local standard of rest used for the survey. One of '0', '1', '2'
	Returns:
		None
	Output:
		Saves the arrays used to make the plots in .npz format, and the figure.
	"""

	xe = np.linspace(-0.9,4.9,250)
	ye = np.linspace(-4.9,16.5,250)
	X,Y = np.meshgrid(xe[:-1],ye[:-1])
	H = [np.zeros_like(X.T), np.zeros_like(X.T),np.zeros_like(X.T)]

	dirname = dirbase+sim+'/lsr_'+nlsr+'/' 
	fname_pattern = dirname+'lsr-'+nlsr+'-rslice-?.'+sim+'-res7100-md-sliced-gcat-dr2.hdf5'
	flist=glob.glob(fname_pattern)
	if len(flist)<1:
		raise RuntimeError('No files match '+ fname_pattern)
	for fn in flist:
		print fn
		with h5py.File(fn,mode='r') as f:
			n = f['l'].shape[0]
			for chunk in range(0,n,dn):
				dc = min(chunk+dn,n)
				print '\tprocessing {0} to {1}'.format(chunk, dc)

				#make cuts on uncertainty and extinction 

				psel = (f['parallax_over_error'][chunk:dc]>10)
				gsel = (f['phot_g_mean_mag_error'][chunk:dc]<0.22)
				bsel = (f['phot_bp_mean_mag_error'][chunk:dc]<0.054)
				rsel = (f['phot_rp_mean_mag_error'][chunk:dc]<0.054)
				ebvsel = (f['a_g_val'][chunk:dc]<0.015)
				sel=(psel&gsel&rsel&bsel&ebvsel)

				#compute and select by tangential velocity

				vt = 4.74 * np.sqrt(f['pmra'][chunk:dc][sel]**2 + f['pmdec'][chunk:dc][sel]**2) / f['parallax'][chunk:dc][sel] 
				vsel=[(vt<40),(vt>60)&(vt<150),(vt>200)]

				#compute color and absolute mag (using 1/parallax)

				color = f['phot_bp_mean_mag'][chunk:dc][sel] - f['phot_rp_mean_mag'][chunk:dc][sel] + f['e_bp_min_rp_val'][chunk:dc][sel]
				dmod = 5.0 * np.log10(100./f['parallax'][chunk:dc][sel])
				mag = f['phot_g_mean_mag'][chunk:dc][sel] - dmod

			#make histograms for each tangential velocity bin
			for i in range(3):
			    h, xe, ye = np.histogram2d(color[vsel[i]],mag[vsel[i]],bins=[xe,ye])
			    H[i]+=h

	#convert to sources per squared mag
	dx = float(np.mean(np.diff(xe)))
	dy = float(np.mean(np.diff(ye)))

	for h in H:
		h/=float(dx*dy)

	savename = dirname+'figures/galactic-hrd-vtan-'+sim+'-lsr-'+nlsr
	print 'Writing to', savename
	for h in H:
		print np.min(h), np.max(h)

	np.savez(savename,
		X=X,
		Y=Y,
		H=H)

	#make plot

	my_cmap=cm.get_cmap('gist_heat')
	my_cmap.set_bad('w')
	my_cmap.set_under('w')

	fig,axs=pl.subplots(1,3,figsize=(7,4),sharey=True,sharex=True)

	for i in range(3):
	    im = axs[i].pcolormesh(X,Y,H[i].T,norm=LogNorm(),cmap=my_cmap)#,vmin=1,vmax=1e3)
	axs[0].set_ylim(16.5,-4.9)
	axs[0].set_xlim(-0.9,4.9)

	axs[0].set_title(r'$V_{T}<40$ km s${}^{-1}$',fontsize=12)
	axs[1].set_title(r'$60<V_{T}<150$ km s${}^{-1}$',fontsize=12)
	axs[2].set_title(r'$V_{T}>200$ km s${}^{-1}$',fontsize=12)

	for i in range(3):
	    axs[i].set_xlabel(u'$G_{BP}-G_{RP}$')
	axs[0].set_ylabel(u'$M_G$')

	fig.tight_layout(w_pad=-0.6)


	fig.subplots_adjust(right=0.84)

	dims = fig.axes[2].get_position().bounds

	cbar_ax = fig.add_axes([0.85, dims[1], 0.03, dims[3]])
	fig.colorbar(im, cax=cbar_ax, label='sources mag${}^{-2}$')

	fig.savefig(savename+'.png',dpi=300)

