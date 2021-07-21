#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import rrtmg as FLUXES
import numpy as np
import netCDF4 as nc
import scipy.interpolate

def fname_correct(fname, months, days):
    for i in months:
        for j in days:
            if "2017{:02d}{:02d}".format(i, j) in fname:
                return True
    return False


# In[2]:

path_retrievals = "/mnt/beegfs/user/phi.richter/DATA_PHD/RRTMG/INPUT/INPUT_FOR_RRTMG/RRTMG_input_nomod"
ssp_ice = "./"
files = sorted(os.listdir(path_retrievals))


# In[3]:


model = FLUXES.RRTMG("/home/phi.richter/SOFTWARE_PHD/run_RRTMG_OOP/rrtmg_lw_v5.00_linux_pgi", "/home/phi.richter/SOFTWARE_PHD/run_RRTMG_OOP/rrtmg_sw_v5.00_linux_pgi", ["./", ssp_ice])

## Profiles
path_profiles = "/home/phi.richter/retrieval_recode/Total_Cloud_Water_retrieval/retrieval/trace_gases/"
z = np.loadtxt(os.path.join(path_profiles, "z.csv"), delimiter=",")
co2 = np.loadtxt(os.path.join(path_profiles, "co2.csv"), delimiter=",")
o3 = np.loadtxt(os.path.join(path_profiles, "o3.csv"), delimiter=",")
ch4 = np.loadtxt(os.path.join(path_profiles, "ch4.csv"), delimiter=",")
n2o = np.loadtxt(os.path.join(path_profiles, "n2o.csv"), delimiter=",")

with nc.Dataset("/mnt/beegfs/user/phi.richter/DATA_PHD/ERA5/ozone.nc", "r") as f:
    o3 = f.variables['o3'][:]
    z_era5 = f.variables['z'][:]/9.80665*1e-3
    lat = np.argmin(np.abs(np.array(f.variables['latitude'][:])-81.95))
    lon = np.argmin(np.abs(np.array(f.variables['longitude'][:])-10.33))

o3_f = scipy.interpolate.interp1d(np.array(z_era5[0,:,12,81]), np.array(o3[0,:,12,81]), fill_value="extrapolate")
o3 = o3_f(np.array(z))
o3_ppmv = 28.9644 / 47.9982 * 1e6 * o3
atm = "HAAA4A4"
#model.plot_atmosphere()
# In[4]:


os.mkdir("rrtmg_run_july")
os.chdir("rrtmg_run_july")
#month = 6
#for day in range(2, 19):
epsilon = 0.99*np.ones(16)
epsilon[np.array([5,6,7,8])] = 0.98
for spec in files:
    if ".nc" in spec and fname_correct(spec, [7], list(range(1, 20))):#"2017{:02d}{:02d}".format(month, day) in spec: 
        try:
            if os.path.exists("/home/phi.richter/RRTMG_Cloudnet/RRTMG_{}.nc".format(spec)):
                continue
            model.read_cloudnet(os.path.join(path_retrievals, spec))
            tg = model.read_trace_gases(z, co2, n2o, ch4, o3_ppmv)
            in_cld_rrtm = model.create_inputfile_cloud()
            input_rrtm = model.create_inputfile_atm_terrestrial(cloud=0, semiss=epsilon, atm=atm)
            model.run_RRTMG_terrestrial(clouds=False)
            input_rrtm = model.create_inputfile_atm_terrestrial(cloud=2, semiss=epsilon, atm=atm)
            model.run_RRTMG_terrestrial(clouds=True)
            input_rrtm = model.create_inputfile_atm_solar(cloud=0, atm=atm)
            model.run_RRTMG_solar(clouds=False)
            input_rrtm = model.create_inputfile_atm_solar(cloud=2, atm=atm)
            model.run_RRTMG_solar(clouds=True)
            model.write_results("/home/phi.richter/RRTMG_Cloudnet/RRTMG_{}.nc".format(spec))
        except Exception:
            continue