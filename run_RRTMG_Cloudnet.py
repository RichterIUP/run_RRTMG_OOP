#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import rrtmg as FLUXES
import numpy as np


# In[2]:


path_retrievals = "/mnt/beegfs/user/phi.richter/DATA_PHD/RRTMG/INPUT/INPUT_FOR_RRTMG/RRTMG_input_nomod"
ssp_ice = "./"
files = sorted(os.listdir(path_retrievals))


# In[3]:


model = FLUXES.RRTMG("/home/phi.richter/RRTMG/RRTMG_LW/build/rrtmg_lw_v5.00_linux_ifort", "/home/phi.richter/RRTMG_SW/RRTMG_SW/build/rrtmg_sw_v5.00_linux_pgi_aerosols", ["./", ssp_ice])


# In[4]:


epsilon = 0.99*np.ones(16)
epsilon[np.array([5,6,7])] = 0.98
for spec in files:
    if ".nc" in spec:
        model.read_cloudnet(os.path.join(path_retrievals, spec))
        model.get_cparam(), model.get_position()
        in_cld_rrtm = model.create_inputfile_cloud()
        input_rrtm = model.create_inputfile_atm_terrestrial(cloud=0, co2_mix=415, semiss=epsilon)
        model.run_RRTMG_terrestrial(clouds=False)
        input_rrtm = model.create_inputfile_atm_terrestrial(cloud=2, co2_mix=415, semiss=epsilon)
        model.run_RRTMG_terrestrial(clouds=True)
        input_rrtm = model.create_inputfile_atm_solar(cloud=0, co2_mix=415)
        model.run_RRTMG_solar(clouds=False)
        input_rrtm = model.create_inputfile_atm_solar(cloud=2, co2_mix=415)
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

