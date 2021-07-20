#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import rrtmg as FLUXES
import numpy as np
import netCDF4 as nc


# In[2]:

month = 6
day = 1
path_retrievals = "/mnt/beegfs/user/phi.richter/DATA_PHD/RRTMG/INPUT/INPUT_FOR_RRTMG/RRTMG_input_nomod"
ssp_ice = "./"
files = sorted(os.listdir(path_retrievals))


# In[3]:


model = FLUXES.RRTMG("./rrtmg_lw_v5.00_linux_pgi", "./rrtmg_sw_v5.00_linux_pgi", ["./", ssp_ice])

## Profiles
z = np.loadtxt("/home/philipp/Doktorandenzeit/SOFTWARE_PHD/Total_Cloud_Water_retrieval/retrieval/trace_gases/z.csv", delimiter=",")
co2 = np.loadtxt("/home/philipp/Doktorandenzeit/SOFTWARE_PHD/Total_Cloud_Water_retrieval/retrieval/trace_gases/co2.csv", delimiter=",")
o3 = np.loadtxt("/home/philipp/Doktorandenzeit/SOFTWARE_PHD/Total_Cloud_Water_retrieval/retrieval/trace_gases/o3.csv", delimiter=",")
ch4 = np.loadtxt("/home/philipp/Doktorandenzeit/SOFTWARE_PHD/Total_Cloud_Water_retrieval/retrieval/trace_gases/ch4.csv", delimiter=",")
n2o = np.loadtxt("/home/philipp/Doktorandenzeit/SOFTWARE_PHD/Total_Cloud_Water_retrieval/retrieval/trace_gases/n2o.csv", delimiter=",")

with nc.Dataset("/mnt/beegfs/user/phi.richter/DATA_PHD/ERA5/ozone.nc", "r") as f:
    o3 = f.variables['o3'][:]
    z_era5 = f.variables['z'][:]/9.80665*1e-3
    lat = np.argmin(np.abs(np.array(f.variables['latitude'][:])-81.95))
    lon = np.argmin(np.abs(np.array(f.variables['longitude'][:])-10.33))

o3_ppmv = 28.9644 / 47.9982 * 1e6 * o3
tg = model.read_trace_gases(z, co2, n2o, ch4, o3_ppmv)
atm = "HAAA4A4"
#model.plot_atmosphere()
# In[4]:


epsilon = 0.99*np.ones(16)
epsilon[np.array([5,6,7,8])] = 0.98
for spec in files:
    if ".nc" in spec and "2017{:02d}{:02d}".format(month, day) in spec: 
        model.read_cloudnet(os.path.join(path_retrievals, spec))
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


# In[5]:


#model.plot_atmosphere()


# In[6]:


#epsilon = 0.99*np.ones(16)
#epsilon[np.array([5,6,7])] = 0.98
#in_cld_rrtm = model.create_inputfile_cloud()
#input_rrtm = model.create_inputfile_atm_terrestrial(cloud=0, co2_mix=415, semiss=epsilon)
#model.run_RRTMG_terrestrial(clouds=False)
#input_rrtm = model.create_inputfile_atm_terrestrial(cloud=2, co2_mix=415, semiss=epsilon)
#model.run_RRTMG_terrestrial(clouds=True)
#input_rrtm = model.create_inputfile_atm_solar(cloud=0, co2_mix=415)
#model.run_RRTMG_solar(clouds=False)
#input_rrtm = model.create_inputfile_atm_solar(cloud=2, co2_mix=415)
#model.run_RRTMG_solar(clouds=True)


# In[7]:


#model.write_results("/home/philipp/out.nc")

