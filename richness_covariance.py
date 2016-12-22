#!/usr/bin/env python

from __future__ import print_function, division

from six.moves import cPickle as pickle
from multiprocessing import Pool

import numpy as np

import astropy.cosmology
from astropy.table import Table
from astropy import units as u

from NFW.nfw import NFW
from colossus.cosmology import cosmology as colossuscosmo
from colossus.halo import concentration

from tqdm import tqdm, trange


# ## Start configurable parameters ## #
#
# Number of parallel processes, set to None for system maximum
processes = None
#
# Location of redMaPPer catalog, must contain keys
# richness_key
# redshift_key
# jk_id_key
catalog_name = "data/y1a1_gold_1.0.3-d10-mof-001b_run_redmapper_v6.4.14-vlim_lgt5_desformat_catalog_JK_IDs.fit"
richness_key = 'LAMBDA_CHISQ'
redshift_key = 'Z_LAMBDA'
jk_id_key = 'JK_ID'

# richness bin edges
richness_edges = [5, 10, 14, 20, 30, 45, 60, np.inf]

# redshift bin edges
z_edges = [0.2, 0.35, 0.5, 0.65]

# \Delta\Sigma profile edges. The bin location should be approximated as the
# area weighted radius
delta_sigma_edges = np.logspace(np.log10(0.02), np.log10(30), 16) * u.Mpc

# cosmology
cosmo = astropy.cosmology.FlatLambdaCDM(70, 0.3)
# ## End configurable parameters ## #


def mass_scaling(lam, z, f_lambda=1.12, g_z=0.18,
                 m0=u.Quantity(10**14.371, u.solMass), lam0=30, z0=0.5):
    """
    The mass-richness scaling relation of Melchior et al. 2017

    Parameters:
    ===========
    lam: array_like, lambda richness values
    z: array_like, cluster redshifts
    f_lambda: float, richness scaling, optional
    g_z: float, redshift scaling, optional
    m0: astropy.units.Quantity, mass pivot, optional

    Returns:
    ========
    m: astropy.units.Quantity, cluster mass
    """
    m = m0 * (lam / lam0)**f_lambda * ((1 + z) / (1 + z0))**g_z
    return m


def read_redMaPPer_catalog(fname):
    tab = Table.read(fname)
    return tab


def compute_concentrations(mass, z):
    cosmo = astropy.cosmology.default_cosmology.get()
    with Pool(processes=processes) as pool:
        results = [pool.apply_async(concentration.modelDiemer15fromM,
                                    [m200.value, z])
                   for m200, z in tqdm(zip(mass * cosmo.h, z),
                                       total=len(mass))]
        c = [results[i].get() for i in trange(len(results))]
    c = np.array(c)
    return c


def get_delta_sigma(m200, c200, zl, r):
    nfw = NFW(m200, c200, zl)
    return nfw.delta_sigma(r)


def compute_delta_sigma_profiles(tab):
    z = tab[redshift_key]
    mass = mass_scaling(tab[richness_key], z)
    print("  Computing cluster concentrations")
    c = compute_concentrations(mass, tab[redshift_key])
    r = np.sqrt((delta_sigma_edges[:-1]**2 + delta_sigma_edges[1:]**2) / 2)
    print("  Computing Delta Sigma profiles")
    delta_sigma_profiles = np.empty((len(tab), r.size))
    with Pool(processes=processes) as pool:
        results = [pool.apply_async(get_delta_sigma, [m200, c200, zl, r])
                   for m200, c200, zl in tqdm(zip(mass, c, z),
                                              total=len(mass))]
        for i in trange(len(results)):
            delta_sigma_profiles[i] = results[i].get()
    return delta_sigma_profiles


def jackknife_clusters(i, j, tab):
    idx = (tab[richness_key] > richness_edges[j]) \
        & (tab[richness_key] < richness_edges[j + 1]) \
        & (tab[redshift_key] > z_edges[i]) \
        & (tab[redshift_key] > z_edges[i + 1])
    thistab = tab[idx]
    print("  {:d} clusters in this bin".format(len(thistab)))
    delta_sigma_profiles = compute_delta_sigma_profiles(thistab)
    jk_ids = np.unique(thistab[jk_id_key])
    delta_sigma_jk = np.empty((jk_ids.size, delta_sigma_profiles.shape[1]))
    for i, jk_id in enumerate(jk_ids):
        idx = thistab[jk_id_key] != jk_id
        delta_sigma_jk[i] = delta_sigma_profiles[idx, :].mean(axis=0)
    delta_sigma_jk_mean = delta_sigma_profiles.mean(axis=0)
    cov = np.cov(delta_sigma_jk - delta_sigma_jk_mean, rowvar=False,
                 bias=True)
    cov *= delta_sigma_jk_mean.size - 1
    return cov


def main():
    astropy.cosmology.default_cosmology.set(cosmo)
    params = {'flat': True, 'H0': cosmo.H0.value,
              'Om0': cosmo.Om0, 'Ob0': 0.046,
              'sigma8': 0.81, 'ns': 0.95}
    colossus_cosmo = colossuscosmo.setCosmology('myCosmo', params)

    tab = read_redMaPPer_catalog(catalog_name)
    for i in range(len(z_edges) - 1):
        for j in range(len(richness_edges) - 1):
            print("Working on richness bin {:d} redshift bin {:d}".format(
                j, i))
            cov = jackknife_clusters(i, j, tab)
            fname = "output/richness_cov_z{:d}_l{:d}.pkl".format(i, j)
            with open(fname, "wb") as f:
                pickle.dump(cov, f)
            break
        break

if __name__ == "__main__":
    main()
