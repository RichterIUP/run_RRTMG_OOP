#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
sys.path.append("/home/phi.richter/SOFTWARE_PHD/run_RRTMG_OOP")
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

path_retrievals = "/mnt/beegfs/user/phi.richter/DATA_PHD/RRTMG/INPUT/INPUT_FOR_RRTMG/{}".format(sys.argv[1])
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

ch4 = ch4 / 1.7 * 1.86

with nc.Dataset("/mnt/beegfs/user/phi.richter/REMOVE/ERA5/ozone.nc", "r") as f:
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

counter = 0
repeat = True
while repeat:
    if not os.path.exists("run_{}_{}".format(sys.argv[1], counter)):
        os.mkdir("run_{}_{}".format(sys.argv[1], counter))
        os.chdir("run_{}_{}".format(sys.argv[1], counter))
        repeat = False
    else:
        counter += 1

if not os.path.exists("/mnt/beegfs/user/phi.richter/REMOVE/OUTPUT/from_rrtmg_oop/{}".format(sys.argv[1])):
    os.mkdir("/mnt/beegfs/user/phi.richter/REMOVE/OUTPUT/from_rrtmg_oop/{}".format(sys.argv[1]))
epsilon = 0.99*np.ones(16)
epsilon[np.array([5,6,7,8])] = 0.98
for spec in files:
    try:
        model.read_cloudnet(os.path.join(path_retrievals, spec), pattern_fname="ERA5_%Y%m%d_%H%M%S.nc")
        model.scale("rliq", 0.75)
        model.scale("rice", 0.75)
        #model.offset("clevel", 1)
        #model.scale("cwp", 1.5)
        tg = model.read_trace_gases(z, co2, n2o, ch4, o3_ppmv)
        in_cld_rrtm = model.create_inputfile_cloud()
        input_rrtm = model.create_inputfile_atm_terrestrial(cloud=0, semiss=epsilon, atm=atm)
        model.run_RRTMG_terrestrial(clouds=False)
        input_rrtm = model.create_inputfile_atm_terrestrial(cloud=2, semiss=epsilon, atm=atm)
        model.run_RRTMG_terrestrial(clouds=True)
        #input_aer = model.create_inputfile_aerosols_solar(level=[1], aot=[0.05], num_aer=1, iaod=0, issa=0, ipha=0, aerpar=[0.13, 1.0, 0.0], ssa=[0.780], phase=[0.7])
        input_rrtm = model.create_inputfile_atm_solar(cloud=0, atm=atm, aerosols=0)
        model.run_RRTMG_solar(clouds=False)
        input_rrtm = model.create_inputfile_atm_solar(cloud=2, atm=atm, aerosols=0)
        model.run_RRTMG_solar(clouds=True)
        #model.write_results("/mnt/beegfs/user/phi.richter/REMOVE/OUTPUT/from_rrtmg_oop/{}/RRTMG_{}_scale_cwp_rliq_rice.nc".format(sys.argv[1], spec))#Scale
        #model.write_results("/mnt/beegfs/user/phi.richter/REMOVE/OUTPUT/from_rrtmg_oop/{}/RRTMG_{}_scale_clevel_1.nc".format(sys.argv[1], spec))#Nomod
        #model.write_results("/mnt/beegfs/user/phi.richter/REMOVE/OUTPUT/from_rrtmg_oop/{}/RRTMG_{}_scale_cwp_15.nc".format(sys.argv[1], spec))#Nomod
        #model.write_results("/mnt/beegfs/user/phi.richter/REMOVE/OUTPUT/from_rrtmg_oop/{}/RRTMG_{}_aot_005.nc".format(sys.argv[1], spec))#Nomod
        model.write_results("/mnt/beegfs/user/phi.richter/REMOVE/OUTPUT/from_rrtmg_oop/{}_scale_rl_075_ri_075/RRTMG_{}.nc".format(sys.argv[1], spec))
    except Exception:
        continue
'''       
    try:
        model.read_cloudnet(os.path.join(path_retrievals, spec))#, pattern_fname="ERA5_%Y%m%d_%H%M%S.nc")
        #model.scale("rliq", 1.5)
        #model.scale("rice", 1.5)
        #model.offset("clevel", 1)
        #model.scale("cwp", 1.5)
        tg = model.read_trace_gases(z, co2, n2o, ch4, o3_ppmv)
        in_cld_rrtm = model.create_inputfile_cloud()
        input_rrtm = model.create_inputfile_atm_terrestrial(cloud=0, semiss=epsilon, atm=atm)
        model.run_RRTMG_terrestrial(clouds=False)
        input_rrtm = model.create_inputfile_atm_terrestrial(cloud=2, semiss=epsilon, atm=atm)
        model.run_RRTMG_terrestrial(clouds=True)
        input_aer = model.create_inputfile_aerosols_solar(level=[1], aot=[0.1], num_aer=1, iaod=0, issa=0, ipha=0, aerpar=[0.13, 1.0, 0.0], ssa=[0.780], phase=[0.7])
        input_rrtm = model.create_inputfile_atm_solar(cloud=0, atm=atm, aerosols=10)
        model.run_RRTMG_solar(clouds=False)
        input_rrtm = model.create_inputfile_atm_solar(cloud=2, atm=atm, aerosols=10)
        model.run_RRTMG_solar(clouds=True)
        #model.write_results("/mnt/beegfs/user/phi.richter/REMOVE/OUTPUT/from_rrtmg_oop/{}/RRTMG_{}_scale_cwp_rliq_rice.nc".format(sys.argv[1], spec))
        #model.write_results("/mnt/beegfs/user/phi.richter/REMOVE/OUTPUT/from_rrtmg_oop/{}/RRTMG_{}_scale_clevel_1.nc".format(sys.argv[1], spec))
        #model.write_results("/mnt/beegfs/user/phi.richter/REMOVE/OUTPUT/from_rrtmg_oop/{}/RRTMG_{}_scale_cwp_15.nc".format(sys.argv[1], spec))
        model.write_results("/mnt/beegfs/user/phi.richter/REMOVE/OUTPUT/from_rrtmg_oop/{}/RRTMG_{}_aot_01.nc".format(sys.argv[1], spec))
    except Exception:
        continue
        
    try:
        model.read_cloudnet(os.path.join(path_retrievals, spec))#, pattern_fname="ERA5_%Y%m%d_%H%M%S.nc")
        #model.scale("rliq", 1.5)
        #model.scale("rice", 1.5)
        #model.offset("clevel", 1)
        #model.scale("cwp", 1.5)
        tg = model.read_trace_gases(z, co2, n2o, ch4, o3_ppmv)
        in_cld_rrtm = model.create_inputfile_cloud()
        input_rrtm = model.create_inputfile_atm_terrestrial(cloud=0, semiss=epsilon, atm=atm)
        model.run_RRTMG_terrestrial(clouds=False)
        input_rrtm = model.create_inputfile_atm_terrestrial(cloud=2, semiss=epsilon, atm=atm)
        model.run_RRTMG_terrestrial(clouds=True)
        input_aer = model.create_inputfile_aerosols_solar(level=[1], aot=[0.15], num_aer=1, iaod=0, issa=0, ipha=0, aerpar=[0.13, 1.0, 0.0], ssa=[0.780], phase=[0.7])
        input_rrtm = model.create_inputfile_atm_solar(cloud=0, atm=atm, aerosols=10)
        model.run_RRTMG_solar(clouds=False)
        input_rrtm = model.create_inputfile_atm_solar(cloud=2, atm=atm, aerosols=10)
        model.run_RRTMG_solar(clouds=True)
        #model.write_results("/mnt/beegfs/user/phi.richter/REMOVE/OUTPUT/from_rrtmg_oop/{}/RRTMG_{}_scale_cwp_rliq_rice.nc".format(sys.argv[1], spec))
        #model.write_results("/mnt/beegfs/user/phi.richter/REMOVE/OUTPUT/from_rrtmg_oop/{}/RRTMG_{}_scale_clevel_1.nc".format(sys.argv[1], spec))
        #model.write_results("/mnt/beegfs/user/phi.richter/REMOVE/OUTPUT/from_rrtmg_oop/{}/RRTMG_{}_scale_cwp_15.nc".format(sys.argv[1], spec))
        model.write_results("/mnt/beegfs/user/phi.richter/REMOVE/OUTPUT/from_rrtmg_oop/{}/RRTMG_{}_aot_015.nc".format(sys.argv[1], spec))
    except Exception:
        continue
'''