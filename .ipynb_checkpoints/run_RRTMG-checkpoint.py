#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:50:11 2021

@author: philipp
"""

import os
import sys
import rrtmg as FLUXES
import numpy as np
import scipy.interpolate
import netCDF4 as nc

os.mkdir(sys.argv[1])
os.chdir(sys.argv[1])

## Profiles
z = np.loadtxt("/home/phi.richter/SOFTWARE_PHD/Total_Cloud_Water_retrieval/retrieval/trace_gases/z.csv", delimiter=",")
co2 = np.loadtxt("/home/phi.richter/SOFTWARE_PHD/Total_Cloud_Water_retrieval/retrieval/trace_gases/co2.csv", delimiter=",")
ch4 = np.loadtxt("/home/phi.richter/SOFTWARE_PHD/Total_Cloud_Water_retrieval/retrieval/trace_gases/ch4.csv", delimiter=",")
n2o = np.loadtxt("/home/phi.richter/SOFTWARE_PHD/Total_Cloud_Water_retrieval/retrieval/trace_gases/n2o.csv", delimiter=",")

## Download Ozone
with nc.Dataset("/home/philipp/ozone.nc", "r") as f:
    o3 = f.variables['o3'][:]
    z_era5 = f.variables['z'][:]/9.80665*1e-3
    lat = np.argmin(np.abs(np.array(f.variables['latitude'][:])-81.95))
    lon = np.argmin(np.abs(np.array(f.variables['longitude'][:])-10.33))
print(lat, lon)

## Interpolate to altitude grid
o3_f = scipy.interpolate.interp1d(np.array(z_era5[0,:,12,81]), np.array(o3[0,:,12,81]), fill_value="extrapolate")
o3 = o3_f(np.array(z))

## Set up model
path_retrievals = "/mnt/beegfs/user/phi.richter/DATA_PHD/RRTMG/INPUT/INPUT_FOR_RRTMG/{}".format(sys.argv[1])
ssp_ice = "/home/phi.richter/SOFTWARE_PHD/Total_Cloud_Water_retrieval/ssp_db.Droxtal.gamma.0p100"
ssp_liq = "/home/phi.richter/SOFTWARE_PHD/Total_Cloud_Water_retrieval/ssp_database/ssp_db.mie_wat.gamma_sigma_0p100"
files = sorted(os.listdir(path_retrievals))
binary_lw = "/home/phi.richter/RRTMG/RRTMG_LW/build/rrtmg_lw_v5.00_linux_ifort"
binary_sw = "/home/phi.richter/RRTMG_SW/RRTMG_SW/build/rrtmg_sw_v5.00_linux_pgi_aerosols"
model = FLUXES.RRTMG(binary_lw, binary_sw, [ssp_liq, ssp_ice])
epsilon = 0.99*np.ones(16)
epsilon[np.array([5,6,7])] = 0.98

for spec in files:
    ## Read input
    model.read_cloudnet(os.path.join(path_retrievals, spec))
    model.get_cparam(), model.get_position()
    model.plot_atmosphere()
    
    ## Input trace gases
    o3_ppmv = 28.9644 / 47.9982 * 1e6 * o3
    tg = model.read_trace_gases(z, co2, n2o, ch4, o3_ppmv)

    ## Run RRTMG
    try:
        in_cld_rrtm = model.create_inputfile_cloud()
        input_rrtm = model.create_inputfile_atm_terrestrial(cloud=0, semiss=epsilon, atm="HAAA4A4")
        model.run_RRTMG_terrestrial(clouds=False)
        input_rrtm = model.create_inputfile_atm_terrestrial(cloud=2, semiss=epsilon, atm="HAAA4A4")
        model.run_RRTMG_terrestrial(clouds=True)
        input_rrtm = model.create_inputfile_atm_solar(cloud=0)
        model.run_RRTMG_solar(clouds=False)
        input_rrtm = model.create_inputfile_atm_solar(cloud=2)
        model.run_RRTMG_solar(clouds=True)
    except Exception:
        continue
    
    ## Write to file
    model.write_results("/home/philipp/RRTMG_{}".format(spec))