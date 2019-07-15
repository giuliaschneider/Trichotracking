import numpy as np
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity
from sklearn import mixture
import scipy.signal

from IPython.core.debugger import set_trace


def gauss(x, mu, sigma):
    return np.exp(-(x-mu)**2 / (2*sigma**2))

def bimodal(x, mu1, sigma1, mu2, sigma2, p):
    return p*gauss(x,mu1,sigma1)+(1-p)*gauss(x,mu2,sigma2)



def fitDistribution(raw, bins, hist):
    if len(x.shape)>1:
        x = x[:,0]
    m = x.mean()
    st = x.std()
    p0 = (m-st, st/2, m+st, st/2, 1)

    params = curve_fit(bimodal, x, freq, p0=p0)
    mu1, sigma1, mu2, sigma2, p = p0
    d = np.abs(mu1 - mu2) / (2*np.sqrt(sigma1 * sigma2))
    if d <= 1:
        set_trace()
        p0 = (m, st)
        mu1, sigma1 = curve_fit(gaussian, x, freq, p0=p0)
        gauss = gaussian(x, mu1, sigma1)
    else:
        gauss = bimodal(x, mu1, sigma1, mu2, sigma2, p)

    return x, gauss

def find_ngaussian(x):
    if len(x.shape)<2:
        x = x[~np.isnan(x), np.newaxis]
    nComponents = np.array([1, 2])
    bic = np.zeros(nComponents.size)
    for i, n in enumerate(nComponents):
        gmm = mixture.GaussianMixture(n_components=n, covariance_type='full')
        gmm.fit(x)
        bic[i] = gmm.bic(x)
    i = np.argmin(bic)
    n = nComponents[i]
    print(n)
    gmm = mixture.GaussianMixture(n_components=n, covariance_type='full')
    gmm.fit(x)
    return gmm

def fit_bigaussian(x):
    gmm = mixture.GaussianMixture(n_components=2, covariance_type='full')
    gmm.fit(x)
    return gmm

def fit_gaussian(x, n_components=1):
    gmm = mixture.GaussianMixture(n_components=n_components,
                                  covariance_type='full')
    gmm.fit(x)
    return gmm

def get_mean(x, kde, xlim):
    # Check if data is bimodal distributed
    xplot = np.linspace(xlim[0], xlim[1], 1000)[:, np.newaxis]
    pdf = kde.score_samples(xplot)
    pe, properties = scipy.signal.find_peaks(pdf, prominence=0.4)
    pe = pe.flatten()
    xplot = xplot.flatten()
    print(xplot[pe])
    if len(pe) <= 1:
        print('unimodal')
        return x.mean()
    elif np.mean(np.diff(xplot[pe]))<x.std():
        print('unimodal')
        return x.mean()
    else:
        xkde = kde.sample(100000)
        n = len(pe)
        gmm = fit_gaussian(xkde, n)
        mus = (gmm.means_).flatten()
        sigmas = (gmm.covariances_).flatten()
        return mus.max()



def fit_kde(raw, xlim, bandwidth=0.1):
    kernel = 'gaussian'
    X = raw[~np.isnan(raw), np.newaxis]
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(X)
    xplot = np.linspace(xlim[0], xlim[1], 1000)[:, np.newaxis]
    log_dens = kde.score_samples(xplot)
    pde = np.exp(log_dens)
    return xplot, pde, kde
