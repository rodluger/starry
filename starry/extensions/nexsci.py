"""Module for nexsci queries"""
import pandas as pd
import numpy as np
import os
import datetime
import astropy.units as u
from glob import glob

from . import PACKAGEDIR
from . import Secondary, Primary, System, Map


class DataRetrievalFailure(Exception):
    pass

def from_nexsci(name, limb_darkening=[0.4, 0.2]):
    """Returns a starry system with properties of system

    Parameters
    ----------
    name : str
        Name of system to retrieve. e.g. 'K2-43'
    """
    df = _get_nexsci_data()
    if np.any([name.endswith(i) for i in 'bcdefghijklmnopqrstuvwxyz']):
        df = df[(df.pl_hostname + df.pl_letter) == name].reset_index(drop=True)
    else:
        df = df[(df.pl_hostname) == name].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("{} does not exist at NExSci".format(name))
    star = Primary(Map(udeg=len(limb_darkening), L=1))
    [star.map[idx + 1] = limb_darkening[idx] for idx in range(len(limb_darkening))]

    planets = []
    for idx, d in df.iterrows():
        planet = kepler.Secondary(starry.Map(L=0),
            r = (np.asarray(df.pl_radj*u.jupiterRad.to(u.solRad)) / np.asarray(df.st_rad))
            tref = d.pl_tranmid
            porb = d.pl_orbper
            inc = d.pl_orbincl)
#        planet.ecc = d.pl_eccen
        planets.append(planet)
    sys = System(star, *planets)
    return sys

def _fill_data(df):
    nan = ~np.isfinite(df.pl_orbincl)
    df.loc[nan, ['pl_orbincl']] = np.zeros(nan.sum()) + 90

    nan = ~np.isfinite(df.pl_eccen)
    df.loc[nan, ['pl_eccen']] = np.zeros(nan.sum())

    # Fill in missing trandep
    nan = ~np.isfinite(df.pl_trandep)
    trandep = (np.asarray(df.pl_radj*u.jupiterRad.to(u.solRad)) / np.asarray(df.st_rad))**2
    df.loc[nan, ['pl_trandep']] = trandep[nan]

    return df

def _retrieve_online_data(guess_masses=False):
    NEXSCI_API = 'http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI'
    try:
        planets = pd.read_csv(NEXSCI_API + '?table=planets&select=pl_hostname,pl_letter,pl_disc,ra,dec,pl_trandep,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_tranflag,pl_trandur,pl_pnum,pl_k2flag,pl_kepflag,pl_facility,pl_orbincl,pl_orbinclerr1,pl_orbinclerr2,pl_orblper,st_mass,st_masserr1,st_masserr2,st_rad,st_raderr1,st_raderr2,st_teff,st_tefferr1,st_tefferr2,st_optmag,st_j,st_h', comment='#')
        composite = pd.read_csv(NEXSCI_API + '?table=compositepars&select=fpl_hostname,fpl_letter,fpl_smax,fpl_smaxerr1,fpl_smaxerr2,fpl_radj,fpl_radjerr1,fpl_radjerr2,fpl_bmassj,fpl_bmassjerr1,fpl_bmassjerr2,fpl_bmassprov,fpl_eqt,fpl_orbper,fpl_orbpererr1,fpl_orbpererr2,fpl_eccen,fpl_eccenerr1,fpl_eccenerr2,fst_spt', comment='#')
        composite.columns = ['pl_hostname','pl_letter','pl_orbsmax','pl_orbsmaxerr1','pl_orbsmaxerr2','pl_radj','pl_radjerr1','pl_radjerr2','pl_bmassj','pl_bmassjerr1','pl_bmassjerr2','pl_bmassprov','pl_eqt','pl_orbper','pl_orbpererr1','pl_orbpererr2', 'pl_eccen','pl_eccenerr1','pl_eccenerr2','st_spt']
    except:
        raise DataRetrievalFailure("Couldn't obtain data from NEXSCI. Do you have an internet connection?")
    df = pd.merge(left=planets, right=composite, how='left', left_on=['pl_hostname', 'pl_letter'],
         right_on = ['pl_hostname', 'pl_letter'])
    df = _fill_data(df)
    df[df.pl_tranflag==1].to_csv("{}/data/planets.csv".format(PACKAGEDIR), index=False)


def _check_data_on_import():
    fname = glob("{}/data/planets.csv".format(PACKAGEDIR))
    if len(fname) == 0:
        _retrieve_online_data()
        fname = glob("{}/data/planets.csv".format(PACKAGEDIR))
    st = os.stat(fname[0])
    mtime = st.st_mtime
    # If database is out of date, get it again.
    if (datetime.datetime.now() - datetime.datetime.fromtimestamp(mtime) > datetime.timedelta(days=7)):
        log.warning('Database out of date. Redownloading...')
        _retrieve_online_data()

def _get_nexsci_data():
    '''Obtain pandas dataframe of all exoplanet data
    '''
    return pd.read_csv("{}/data/planets.csv".format(PACKAGEDIR))

_check_data_on_import()
