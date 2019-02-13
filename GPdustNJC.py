
import sys
if sys.path[0] != '/mnt/home/landerson/.local/lib/python3.6/site-packages':
    sys.path.insert(0, '/mnt/home/landerson/.local/lib/python3.6/site-packages/astroML-0.3-py3.6.egg')
    sys.path.insert(0, '/mnt/home/landerson/.local/lib/python3.6/site-packages/xdgmm-1.0.9-py3.6.egg')
    sys.path.insert(0, '/mnt/home/landerson/.local/lib/python3.6/site-packages')
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
import numpy as np
import matplotlib as mpl
from scipy import stats
from dustmaps.sfd import SFDQuery
from dustmaps.bayestar import BayestarQuery
import astropy.coordinates as coord
import astropy.units as u
import scipy.stats
from astropy.table import Table, unique, Column, hstack, vstack
import healpy as hp
from xdgmm import XDGMM

import scipy.optimize as op
import emcee
import corner


def dust(ra, dec, distance, max_samples=10, mode='median'):
    c = coord.SkyCoord(ra, dec, distance=distance)
    sfd = SFDQuery()
    bayes = BayestarQuery(max_samples=max_samples)

    return sfd(c), bayes(c, mode=mode, return_flags=False) #, iphas(c, mode=mode), marshall(c), chen(c)

def getDust(G, bp, rp, ebv, maxnit=100):
    """ Compute the Gaia extinctions assuming relations from Babusieux
    Arguments: G, bp, rp, E(B-V)
    maxnit -- number of iterations
    Returns extinction in G,bp, rp
    Author: Sergey Koposov skoposov@cmu.edu
    """
    c1, c2, c3, c4, c5, c6, c7 = [0.9761, -0.1704,
                                  0.0086, 0.0011, -0.0438, 0.0013, 0.0099]
    d1, d2, d3, d4, d5, d6, d7 = [
        1.1517, -0.0871, -0.0333, 0.0173, -0.0230, 0.0006, 0.0043]
    e1, e2, e3, e4, e5, e6, e7 = [
        0.6104, -0.0170, -0.0026, -0.0017, -0.0078, 0.00005, 0.0006]
    A0 = 3.1*ebv
    P1 = np.poly1d([c1, c2, c3, c4][::-1])

    def F1(bprp): return np.poly1d(
        [c1, c2, c3, c4][::-1])(bprp)+c5*A0+c6*A0**2+c7*bprp*A0

    def F2(bprp): return np.poly1d(
        [d1, d2, d3, d4][::-1])(bprp)+d5*A0+d6*A0**2+d7*bprp*A0

    def F3(bprp): return np.poly1d(
        [e1, e2, e3, e4][::-1])(bprp)+e5*A0+e6*A0**2+e7*bprp*A0
    xind = np.isfinite(bp+rp+G)
    curbp = bp-rp
    for i in range(maxnit):
        AG = F1(curbp)*A0
        Abp = F2(curbp)*A0
        Arp = F3(curbp)*A0
        curbp1 = bp-rp-Abp+Arp

        delta = np.abs(curbp1-curbp)[xind]
        curbp = curbp1
    print(scipy.stats.scoreatpercentile(delta[np.isfinite(delta)], 99))
    AG = F1(curbp)*A0
    Abp = F2(curbp)*A0
    Arp = F3(curbp)*A0
    return AG, Abp, Arp

def matrixize(data1, data2, err1, err2):
    """
    vectorize the 2 pieces of data into a 2D mean and 2D covariance matrix
    """
    X = np.vstack([data1, data2]).T
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:, diag, diag] = np.vstack([err1**2., err2**2.]).T
    return X, Xerr

def plotXdgmm(xdgmm, ax, c='k', lw=1, label='prior', step=0.001):
    ts = np.arange(0, 2. * np.pi, step) #magic
    amps = xdgmm.weights
    mus = xdgmm.mu
    Vs = xdgmm.V
    for gg in range(xdgmm.n_components):
        if amps[gg] == np.max(amps):
            label=label
        else:
            label=None
        w, v = np.linalg.eigh(Vs[gg])
        points = np.sqrt(w[0]) * (v[:, 0])[:,None] * (np.cos(ts))[None, :] +                  np.sqrt(w[1]) * (v[:, 1])[:,None] * (np.sin(ts))[None, :] +                  mus[gg][:, None]
        ax.plot(points[0,:], points[1,:], c, lw=lw, alpha=amps[gg]/np.max(amps), rasterized=True, label=label)

def lnprior(theta, xdgmm, I=np.zeros((1, 2, 2))):
    hw2_model, mw2_model, Ak = theta
    if (-5 <= Ak) & (Ak<=5):
        return xdgmm.score_samples(np.vstack([hw2_model, mw2_model]).T, I)[0]
    return -np.inf

def lnprob(theta, Y, Y_err, xdgmm):
    lp = lnprior(theta, xdgmm)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, Y, Y_err)

def lnlike(theta, Y, Y_err):
    Aw2_Ak = 0.43
    (hw2_model, mw2_model, Ak) = theta
    Y_model = np.vstack([hw2_model + Ak/0.918, mw2_model + Ak*Aw2_Ak]).T
    inv_sigma_2 = np.linalg.inv(Y_err**2)
    result = -0.5*(np.sum(np.dot(np.dot(Y - Y_model, inv_sigma_2[0]), np.transpose(Y - Y_model)) +
                          np.log(np.linalg.det(Y_err[0]**2))))
    return result


nll = lambda *args: -lnlike(*args)
nlp = lambda *args: -lnprob(*args)

def get_lowdust_arrays():
    try:
        data = np.load('highlatStars.npz')
        return data['color'], data['absmag'], data['colorErr'], data['absmagErr'], data['ra'], data['dec'], data['parallax']
    except IOError:
        datahigh = Table.read('dustHighLat-result.fits.gz')
        datalow = Table.read('dustLowLat-result.fits.gz')
        data = vstack((datahigh, datalow))

        c = coord.SkyCoord(data['ra'], data['dec'], distance=1./data['parallax']/u.mas*u.kpc)
        galc = c.transform_to(coord.Galactic)
        galactic = c.transform_to(coord.Galactocentric)
        highlat = np.abs(galc.b) > 45*u.deg

        absmag = data['w2mpro'] - 5.*np.log10(1./(data['parallax']/1e2))
        color = data['h_m'] - data['w2mpro']

        sfddust, bayesdust = dust(data['ra'], data['dec'], 1./data['parallax']/u.mas*u.kpc, max_samples=10,
                                  mode='median')

        colorErr = np.sqrt(data['h_msigcom']**2 + data['w2mpro_error']**2.)
        absmagErr = data['w2mpro_error']
        indices = highlat & (absmag < 2) & (sfddust < 0.05)
        print(np.sum(indices))
        np.savez('highlatStars', 
                 color=color[indices],
                 absmag=absmag[indices],
                 colorErr = colorErr[indices],
                 absmagErr = absmagErr[indices],
                 ra = data['ra'][indices], 
                 dec = data['dec'][indices],
                 parallax = data['parallax'][indices])
        return color[indices], absmag[indices], colorErr[indices], absmagErr[indices], data['ra'][indices], data['dec'][indices], data['parallax'][indices]

def get_arrays():
    try:
        data = np.load('dustStars.npz')
        return data['color'], data['absmag'], data['colorErr'], data['absmagErr'], data['ra'], data['dec'], data['parallax']
    except IOError:
        datahigh = Table.read('dustHighLat-result.fits.gz')
        datalow = Table.read('dustLowLat-result.fits.gz')
        data = vstack((datahigh, datalow))

        absmag = data['w2mpro'] - 5.*np.log10(1./(data['parallax']/1e2))
        color = data['h_m'] - data['w2mpro']

        colorErr = np.sqrt(data['h_msigcom']**2 + data['w2mpro_error']**2.)
        absmagErr = data['w2mpro_error']
        indices = absmag < 2
        print(np.sum(indices))
        np.savez('dustStars',
                 color=color[indices],
                 absmag=absmag[indices],
                 colorErr = colorErr[indices],
                 absmagErr = absmagErr[indices],
                 ra = data['ra'][indices],
                 dec = data['dec'][indices],
                 parallax = data['parallax'][indices])
        return color[indices], absmag[indices], colorErr[indices], absmagErr[indices], data['ra'][indices], data['dec'][indices], data['parallax'][indices]
    

    

if __name__ == '__main__':

    ncomp = 64
    try:
        xdgmm = XDGMM(filename='rjce_lowdust_{0}G.fits'.format(ncomp))
    except IOError:
        xdgmm = XDGMM(method='Bovy')
        xdgmm.n_components = ncomp
        color, absmag, colorErr, absmagErr, ra, dec, parallax = get_lowdust_arrays()
        X, Xerr = matrixize(color, absmag, colorErr, absmagErr)
        xdgmm = xdgmm.fit(X, Xerr)
        xdgmm.save_model('rjce_lowdust_{0}G.fits'.format(ncomp))

    m = np.int(sys.argv[1])
    n = np.int(sys.argv[2])
    nthreads = np.int(sys.argv[3])

    print(m, n, nthreads)
    hw2 = np.zeros((n-m, 3))
    mw2 = np.zeros((n-m, 3))
    Ak  = np.zeros((n-m, 3))
    
    color, absmag, colorErr, absmagErr, ra, dec, parallax = get_arrays()

    filename = 'posterior_estimates_{0}_{1}.txt'.format(m, n)
    f = open(filename, 'w+')
    f.write('hw2    hw2_high_sigma    hw2_low_sigma    mw2    mw2_high_sigma    mw2_low_sigma    Ak    Ak_high_sigma  Ak_low_sigma    ra    dec    parallax \n')
    f.close()
    for i in np.arange(m, n):
        print(i)
        Y, Y_err = matrixize(color[i],
                             absmag[i],
                             colorErr[i],
                             absmagErr[i])
        Ak_guess = 0.01
        hw2_guess = color[i]
        mw2_guess = absmag[i]
        result = op.minimize(nlp, [hw2_guess, mw2_guess, Ak_guess],
                     args=(Y, Y_err, xdgmm))
        
        ndim, nwalkers = 3, 50
        pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        
        print('starting sampler')
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(Y, Y_err, xdgmm), threads=nthreads)
        sampler.run_mcmc(pos, 200)
        samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
        if sampler.pool is not None: sampler.pool.close()

        hw2_mcmc, mw2_mcmc, ak_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                                          zip(*np.percentile(samples, [16, 50, 84],
                                                             axis=0)))
        hw2[i-m] = hw2_mcmc
        mw2[i-m] = mw2_mcmc
        Ak[i-m] = ak_mcmc
        f = open(filename, 'a')
        f.write('{0:.5f}  {1:0.5f}  {2:0.5f}  {3:0.5f}  {4:0.5f}  {5:0.5f}  {6:0.5f}  {7:0.5f}  {8:0.5f}  {9:0.5f}  {10:0.5f}  {11:0.5f} \n'.format(hw2_mcmc[0], hw2_mcmc[1], hw2_mcmc[2], mw2_mcmc[0], mw2_mcmc[1], mw2_mcmc[2], ak_mcmc[0], ak_mcmc[1], ak_mcmc[2], ra[i], dec[i], parallax[i])) 
        f.close()
    np.savez('posteriorSamples_{0}_{1}'.format(n, m), hw2=hw2, mw2=mw2, Ak=Ak, ra=ra, dec=dec, parallax=parallax)

    #print(hw2, mw2, Ak)
