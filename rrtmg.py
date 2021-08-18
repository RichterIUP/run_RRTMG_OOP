#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:26:11 2021

@author: philipp
"""

import os
import sys
import subprocess
import pandas as pd
import netCDF4 as nc
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import read_database as mie

def read_results(layer, wavenumbers, fname="OUTPUT_RRTM"):
    '''
    Read output files of RRTMG and save all data
    '''
    with open(fname, "r") as f:
        cont = f.readlines()
        
    header = []
    for line in cont:
      
        ## find header
        if "LEVEL" in line:
            for element in line.split("  "):
                if len(element) > 0:
                    if element.rstrip().lstrip() == "LEVEL PRESSURE":
                        text = "LEVEL"
                        if text not in header: header.append(text)
                        text = "PRESSURE"
                        if text not in header: header.append(text)
                    else:
                        text = element.rstrip().lstrip()
                        if text not in header: header.append(text)
                        
    start = -1
    for ii in range(1, len(cont)):
        #print(cont[ii])
        #if "LEVEL" in cont[ii]:
        #    print(str(wavenumbers[0]), str(wavenumbers[1]), cont[ii-1], str(wavenumbers[0]) in cont[ii-1])
        if "LEVEL" in cont[ii] and str(int(wavenumbers[0])) in cont[ii-1] and str(int(wavenumbers[1])) in cont[ii-1]:
            start = ii+2
        if " 0 " in cont[ii] and start != -1:
            end = ii+1
            break
    col = [[] for ii in range(len(header))]

    for line in cont[start:end]:
        ii = 0
        for element in line.split(" "):
            if len(element) > 0:
                if "*" in element:
                    col[ii].append(0)
                else:
                    col[ii].append(float(element))
                ii += 1
    col = np.array(col)
    out = dict()
    for key in range(len(header)):
        out.update({header[key]: col[key, :]})
    out = pd.DataFrame(out)
    return out


class RRTMG:
    def __init__(self, binary_lw, binary_sw, mie_db):
        self.__cloud = {'date': None, \
                        'wavenumber': None, \
                        'spectrum': None, \
                        'latitude': None, \
                        'longitude': None, \
                        'height_prof': None, \
                        'pressure_prof': None, \
                        'humidity_prof': None, \
                        'temperature_prof': None, \
                        'sza': None, \
                        'red_chi_2': None, \
                        'tau_liq': None, \
                        'tau_ice': None, \
                        'rliq': None, \
                        'rice': None, \
                        'clevel': None, \
                        'lwp': None, \
                        'iwp': None, \
                        'wpi': None, \
                        'cwp': None, \
                        'clt': None, \
                        'iceconc': None, \
                        'albedo': None, \
                        'co2': None, \
                        'n2o': None, \
                        'ch4': None, \
                        'o3': None, \
                        'diff_lwp': None, \
                        'flag_lwc': None, \
                        'flag_iwc': None, \
                        'flag_reff': None, \
                        'flag_reffice': None, \
                        'fluxes_sw_all': None, \
                        'fluxes_sw_clear': None, \
                        'fluxes_lw_all': None, \
                        'fluxes_lw_clear': None}
        self.__wn_lw = [[10., 350.], \
                        [350., 500.], \
                        [500., 630.], \
                        [630., 700.], \
                        [700., 820.], \
                        [820., 980.], \
                        [980., 1080.], \
                        [1080., 1180.], \
                        [1180., 1390.], \
                        [1390., 1480.], \
                        [1480., 1800.], \
                        [1800., 2080.], \
                        [2080., 2250.], \
                        [2250., 2380.], \
                        [2380., 2600.], \
                        [2600., 3250.], \
                        [10., 3250.]]
        self.__wn_sw = [[2600., 3250.], \
                        [3250., 4000.], \
                        [4000., 4650.], \
                        [4650., 5150.], \
                        [5150., 6150.], \
                        [6150., 7700.], \
                        [7700., 8050.], \
                        [8050., 12850.], \
                        [12850., 16000.], \
                        [16000., 22650.], \
                        [22650., 29000.], \
                        [29000., 38000.], \
                        [38000., 50000.], \
                        [820., 2600.], \
                        [820., 50000.]]
        self.__binary_lw = binary_lw
        self.__binary_sw = binary_sw
        try:
            [self.__liq_db, self.__ice_db] = mie.read_databases(mie_db[0], mie_db[1])
        except Exception:
            pass
        
    def read_cloudnet(self, fname, pattern_fname="in_CNET_%Y%m%d_%H%M%S"):
        pattern = fname.split("/")[-1][:23]
        self.__cloud['date'] = dt.datetime.strptime(pattern, pattern_fname)
        
        with nc.Dataset(fname, "r") as f:
            self.__cloud['latitude'] = f.variables['lat'][0]
            self.__cloud['longitude'] = f.variables['lon'][0]
            self.__cloud['sza'] = f.variables['sza'][0]
            self.__cloud['cwp'] = f.variables['cwp'][:]
            self.__cloud['wpi'] = f.variables['wpi'][:]
            self.__cloud['lwp'] = self.__cloud['cwp'] * (1-self.__cloud['wpi'])
            self.__cloud['iwp'] = self.__cloud['cwp'] * self.__cloud['wpi']
            self.__cloud['rliq'] = f.variables['rl'][:]
            self.__cloud['rice'] = f.variables['ri'][:]
            self.__cloud['clevel'] = f.variables['cloud'][:]
            self.__cloud['height_prof'] = np.array(f.variables['z'][:])
            self.__cloud['temperature_prof'] = f.variables['t'][:]
            self.__cloud['humidity_prof'] = f.variables['q'][:]
            self.__cloud['pressure_prof'] = f.variables['p'][:]
            self.__cloud['albedo'] = np.float(f.variables['albedo_dir'][0])
            self.__cloud['clt'] = np.ones(self.__cloud['lwp'].size)
            self.__cloud['iceconc'] = np.float(f.variables['iceconc'][0])
            self.__cloud['co2'] = np.abs(f.variables['co2'][:])
            self.__cloud['n2o'] = np.abs(f.variables['n2o'][:])
            self.__cloud['ch4'] = np.abs(f.variables['ch4'][:])
            self.__cloud['diff_lwp'] = f.variables['diff_lwp'][0]
            self.__cloud['flag_lwc'] = f.variables['flag_lwc'][0]
            self.__cloud['flag_iwc'] = f.variables['flag_iwc'][0]
            self.__cloud['flag_reff'] = f.variables['flag_reff'][0]
            self.__cloud['flag_reffice'] = f.variables['flag_reffice'][0]
        self.__cloud['fluxes_sw_all'] = [None for i in range(15)]
        self.__cloud['fluxes_sw_clear'] = [None for i in range(15)]
        self.__cloud['fluxes_lw_all'] = [None for i in range(17)]
        self.__cloud['fluxes_lw_clear'] = [None for i in range(17)]

    def read_tcwret(self, fname, input_radiance=""):
        pattern = fname.split("/")[-1]
        self.__cloud['date'] = dt.datetime.strptime(pattern, "results_%Y%m%d%H%M%S.nc")
        time_dec = self.__cloud['date'].hour + self.__cloud['date'].minute/60.0 + self.__cloud['date'].second/3600.0

        ## Read radiances from inputfiles of TCWret. This is done because the radiance output from TCWret is averaged
        ## for the microwindows. One can use the radiance output from TCWret if no path to the inputfiles is specified
        if input_radiance != "":
            with nc.Dataset(os.path.join(input_radiance, "PS.{:04d}{:02d}{:02d}.nc".format(self.__cloud['date'].year, self.__cloud['date'].month, self.__cloud['date'].day))) as f:
                idx = np.argmin(np.abs(f.variables['time_dec'][:]-time_dec))
                self.__cloud['wavenumber'] = f.variables['wavenumber'][idx]
                self.__cloud['spectrum'] = f.variables['radiance'][idx]
        
        with nc.Dataset(fname, "r") as f:
            self.__cloud['latitude'] = np.float(f.variables['lat'][0])
            self.__cloud['longitude'] = np.float(f.variables['lon'][0])
            self.__cloud['height_prof'] = f.variables['z'][:]*1e-3
            self.__cloud['pressure_prof'] = f.variables['P'][:]
            self.__cloud['humidity_prof'] = f.variables['humidity'][:]
            self.__cloud['temperature_prof'] = f.variables['T'][:]
            self.__cloud['sza'] = np.float(f.variables['sza'][0])
            if input_radiance == "":
                self.__cloud['wavenumber'] = f.variables['wavenumber'][:]
                self.__cloud['spectrum'] = f.variables['lbldis radiance'][:]
        
            self.__cloud['red_chi_2'] = f.variables['red_chi_2'][0]
            xret = f.variables['x_ret'][:]
            self.__cloud['clevel'] = np.array(np.where(self.__cloud['height_prof'] == f.variables['clevel'][:]*1e-3)[0])
            self.__cloud['tau_liq'] = f.variables['x_ret'][0]
            self.__cloud['tau_ice'] = f.variables['x_ret'][1]
            self.__cloud['rliq'] = np.array([np.exp(f.variables['x_ret'][2])])
            self.__cloud['rice'] = np.array([np.exp(f.variables['x_ret'][3])])
            self.__cloud['lwp'] = np.array([mie.calc_lwp(self.__cloud['rliq'], 0.0, self.__cloud['tau_liq'], 0.0)[0]])
            self.__cloud['iwp'] = np.array([mie.calc_iwp(self.__cloud['tau_ice'], 0.0, self.__cloud['rice'], 0.0, self.__ice_db)[0]])
            self.__cloud['cwp'] = self.__cloud['lwp'] + self.__cloud['iwp']
            self.__cloud['wpi'] = self.__cloud['iwp']/self.__cloud['cwp']
            self.__cloud['clt'] = np.ones(self.__cloud['clevel'].size)
            self.__cloud['albedo'] = 0.99
            self.__cloud['co2'] = np.ones(self.__cloud['height_prof'].size)
            self.__cloud['n2o'] = -1*np.ones(self.__cloud['height_prof'].size)
            self.__cloud['ch4'] = -1*np.ones(self.__cloud['height_prof'].size)
            self.__cloud['diff_lwp'] = 0.0
            self.__cloud['flag_lwc'] = 0
            self.__cloud['flag_iwc'] = 0
            self.__cloud['flag_reff'] = 0
            self.__cloud['flag_reffice'] = 0
        self.__cloud['fluxes_sw_all'] = [None for i in range(15)]
        self.__cloud['fluxes_sw_clear'] = [None for i in range(15)]
        self.__cloud['fluxes_lw_all'] = [None for i in range(17)]
        self.__cloud['fluxes_lw_clear'] = [None for i in range(17)]
        
    def scale(self, key, factor):
        self.__cloud[key] = self.__cloud[key]*factor
        
    def offset(self, key, offs):
        self.__cloud[key] = self.__cloud[key]+offs
        
    def replace_atmosphere(self, fname, key_height="height(m)", key_pressure="pressure(hPa)", key_humidity="humidity(%)", key_temperature="temperature(K)"):
        if os.path.exists(fname):
            atm_new = pd.read_csv(fname)
            temp_f = scipy.interpolate.interp1d(atm_new[key_height]*1e-3, atm_new[key_temperature], fill_value="extrapolate")
            humd_f = scipy.interpolate.interp1d(atm_new[key_height]*1e-3, atm_new[key_humidity], fill_value="extrapolate")
            plev_f = scipy.interpolate.interp1d(atm_new[key_height]*1e-3, atm_new[key_pressure], fill_value="extrapolate")
            self.__cloud['temperature_prof'] = temp_f(self.__cloud['height_prof'])
            self.__cloud['humidity_prof'] = humd_f(self.__cloud['height_prof'])
            self.__cloud['pressure_prof'] = plev_f(self.__cloud['height_prof'])
            self.__cloud['temperature_prof'][self.__cloud['temperature_prof'] < 0.0] = 0.0
            self.__cloud['humidity_prof'][self.__cloud['humidity_prof'] < 0.0] = 0.0
            self.__cloud['pressure_prof'][self.__cloud['pressure_prof'] < 0.0] = 0.0

    def plot_spectrum(self, ylim=[-1, -1], xlim=[-1, -1], fname=""):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(self.__cloud['wavenumber'], self.__cloud['spectrum'])
        ax.grid(True)
        if ylim != [-1, -1]: ax.set_ylim(ylim)
        if xlim != [-1, -1]: ax.set_xlim(xlim)
        if fname != "": plt.savefig(fname)
        plt.show()
            
    def plot_atmosphere(self, fname="", ylim=[[0, 50], [1013, 0]]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title("Height coordinates")
        ax[1].set_title("Pressure coordinates")
        ax[0].plot(self.__cloud['humidity_prof'], self.__cloud['height_prof'], label="Humidity")
        ax[0].plot(self.__cloud['temperature_prof'], self.__cloud['height_prof'], label="Temperature")
        ax[0].hlines(self.__cloud['height_prof'][self.__cloud['clevel']], 0, 300, label="Cloud")
        ax[1].plot(self.__cloud['humidity_prof'], self.__cloud['pressure_prof'], label="Humidity")
        ax[1].plot(self.__cloud['temperature_prof'], self.__cloud['pressure_prof'], label="Temperature")
        ax[1].hlines(self.__cloud['pressure_prof'][self.__cloud['clevel']], 0, 300, label="Cloud")
        ax[0].set_ylim(ylim[0])
        ax[1].set_ylim(ylim[1])
        for i in range(2):
            ax[i].grid(True)
            ax[i].legend()
        if fname != "": plt.savefig(fname)
        plt.show()

        
    def temp_half_levels(self):
        t_half = np.zeros(self.__cloud['temperature_prof'].size)
        for i in enumerate(self.__cloud['temperature_prof']):
            if i[0] == self.__cloud['temperature_prof'].size-1: break
            plev_12 = 1/2*(self.__cloud['pressure_prof'][i[0]]+self.__cloud['pressure_prof'][i[0]+1])
            frac_1 = (self.__cloud['pressure_prof'][i[0]]*(self.__cloud['pressure_prof'][i[0]+1]-plev_12))/(plev_12*(self.__cloud['pressure_prof'][i[0]+1]-self.__cloud['pressure_prof'][i[0]]))
            frac_2 = self.__cloud['pressure_prof'][i[0]+1]*(plev_12 - self.__cloud['pressure_prof'][i[0]])/(plev_12*(self.__cloud['pressure_prof'][i[0]+1]-self.__cloud['pressure_prof'][i[0]]))
            t_half[i[0]] = self.__cloud['temperature_prof'][i[0]] * frac_1 + self.__cloud['temperature_prof'][i[0]+1] * frac_2
        t_half[-1] = 1/2*(t_half[-2]+self.__cloud['temperature_prof'][-1])
        self.__cloud['temperature_prof'][1:] = t_half[1:]
        return self.__cloud['temperature_prof']
    
    def read_trace_gases(self, height, co2, n2o, ch4, o3):
        co2_f = scipy.interpolate.interp1d(height, co2, fill_value="extrapolate")
        self.__cloud['co2'] = co2_f(self.__cloud['height_prof'])
        self.__cloud['co2'][self.__cloud['co2'] < 0.0] = 0.0
        n2o_f = scipy.interpolate.interp1d(height, n2o, fill_value="extrapolate")
        self.__cloud['n2o'] = n2o_f(self.__cloud['height_prof'])
        self.__cloud['n2o'][self.__cloud['n2o'] < 0.0] = 0.0
        ch4_f = scipy.interpolate.interp1d(height, ch4, fill_value="extrapolate")
        self.__cloud['ch4'] = ch4_f(self.__cloud['height_prof'])
        self.__cloud['ch4'][self.__cloud['ch4'] < 0.0] = 0.0
        o3_f = scipy.interpolate.interp1d(height, o3, fill_value="extrapolate")
        self.__cloud['o3'] = o3_f(self.__cloud['height_prof'])
        self.__cloud['o3'][self.__cloud['o3'] < 0.0] = 0.0
        return {'height': self.__cloud['height_prof'], \
                'pressure': self.__cloud['pressure_prof'], \
                'co2': self.__cloud['co2'], \
                'n2o': self.__cloud['n2o'], \
                'ch4': self.__cloud['ch4'], \
                'o3': self.__cloud['o3']}

    def create_inputfile_atm_solar(self, aerosols=0, cloud=0, tprof=[-1], pprof=[-1], hprof=[-1], zprof=[-1], albedo=[0.99], atm="4444444", use_albedo_par=False):

        temperature_prof = self.__cloud['temperature_prof'] if tprof[0] == -1 else tprof
        height_prof = self.__cloud['height_prof'] if zprof[0] == -1 else zprof
        humidity_prof = self.__cloud['humidity_prof'] if hprof[0] == -1 else hprof
        pressure_prof = self.__cloud['pressure_prof'] if pprof[0] == -1 else pprof
        
        RECORD_1_1 = "{:80s}".format("$ RRTM_SW runscript created on {}".format(dt.datetime.now()))

        IAER = aerosols
        IATM = 1
        ISCAT = 1
        IOUT = 98
    
        if cloud == 0:
            ICMA = 0
        else:
            ICMA = 1
        ICLD = cloud
        IDELM = 0
        ICOS = 0
    
        RECORD_1_2  = 18 * " " + "{:2d}".format(IAER)
        RECORD_1_2 += 29 * " " + "{:1d}".format(IATM)
        RECORD_1_2 += 32 * " " + "{:1d}".format(ISCAT)
        RECORD_1_2 += 1  * " " + " "#"{:1d}".format(ISTRM)
        RECORD_1_2 += 2  * " " + "{:3d}".format(IOUT)
        RECORD_1_2 += 3  * " " + "{:1d}".format(ICMA)
        RECORD_1_2 += 0  * " " + "{:1d}".format(ICLD)
        RECORD_1_2 += 3  * " " + "{:1d}".format(IDELM)
        RECORD_1_2 += "{:1d}".format(ICOS)

        JULDAT = int((self.__cloud['date'] - dt.datetime(self.__cloud['date'].year, 1, 1)).total_seconds()/(24*3600.0))
        SZA = self.__cloud['sza']
        ISOLVAR = 0
        SCON = 1370
        SOLCYCFRAC = 0
        RECORD_1_2_1  = 12 * " " + "{:3d}".format(JULDAT)
        RECORD_1_2_1 += 3  * " " + "{:7.4f}".format(SZA)
        RECORD_1_2_1 += 3  * " " + "{:2d}".format(ISOLVAR)
        RECORD_1_2_1 += "{:10.4f}".format(SCON)
        RECORD_1_2_1 += "{:10.5f}".format(SOLCYCFRAC)
        IEMIS = 1 if len(albedo) == 1 else 2
        IREFLECT = 0
        if use_albedo_par:
            SEMISS = albedo
        else:
            SEMISS = [self.__cloud['albedo']]
    
        RECORD_1_4  = 11 * " " + "{:1d}".format(IEMIS)
        RECORD_1_4 += 2  * " " + "{:1d}".format(IREFLECT)
        for element in SEMISS:
            RECORD_1_4 += "{:5.3f}".format(1-element)

        MODEL = 0
        IBMAX = len(height_prof)
        NOPRINT = 0
        NMOL = 7
        IPUNCH = 1
        MUNITS = 1
        RE = 0
        REF_LAT = 0
    
        RECORD_3_1  = "{:5d}".format(MODEL)
        RECORD_3_1 += 5 * " " + "{:5d}".format(IBMAX) 
        RECORD_3_1 += 5 * " " + "{:5d}".format(NOPRINT)
        RECORD_3_1 += "{:5d}".format(NMOL)
        RECORD_3_1 += "{:5d}".format(IPUNCH)
        RECORD_3_1 += 3 * " " + "{:2d}".format(MUNITS)
        RECORD_3_1 += "{:10.3f}".format(RE)
        RECORD_3_1 += 20 * " " + "{:10.3f}".format(410)
        RECORD_3_1 += "{:10.3f}".format(REF_LAT)

        HBOUND = height_prof[0]
        HTOA = height_prof[-1]
    
        RECORD_3_2 = "{:10.3f}{:10.3f}".format(HBOUND, HTOA)

        ZBND = ""
        for ii in range(len(height_prof)):
            ZBND += "{:10.3f}".format(height_prof[ii])
            if ii % 8 == 7:
                ZBND += "\n"
    
        RECORD_3_3B = ZBND

        IMMAX = len(height_prof)
        HMOD = ""
        ZM = height_prof
        PM = pressure_prof
        TM = temperature_prof
        JCHARP = "A"
        JCHART = "A"
        JCHAR = atm
        RECORD_3_5_6 = ""
        VOL = np.array([np.zeros(len(height_prof)) for ii in range(7)])
        VOL[0] = humidity_prof
        VOL[1] = self.__cloud['co2']
        VOL[2] = self.__cloud['o3']
        VOL[3] = self.__cloud['n2o']
        VOL[5] = self.__cloud['ch4']
        for loop in range(len(height_prof)):
            RECORD_3_5_6 += "{:10.3E}".format(ZM[loop])
            RECORD_3_5_6 += "{:10.3E}".format(PM[loop])
            RECORD_3_5_6 += "{:10.3f}".format(TM[loop])
            RECORD_3_5_6 += 5 * " " + "{:1s}".format(JCHARP) + "{:1s}".format(JCHART)
            RECORD_3_5_6 += 3 * " " + "{}".format(JCHAR)
            RECORD_3_5_6 += "\n"
            for molecules in range(7):
                RECORD_3_5_6 += "{:10.3E}".format(VOL[molecules][loop])
            RECORD_3_5_6 += "\n"
        
        RECORD_3_4 = "{:5d}".format(IMMAX) + "{:24s}".format(HMOD)
        ret = RECORD_1_1 + "\n" + RECORD_1_2 + "\n" + RECORD_1_2_1 + "\n" + RECORD_1_4 + "\n" + \
            RECORD_3_1 + "\n" + RECORD_3_2 + "\n" + RECORD_3_3B + "\n" + RECORD_3_4 + "\n" + \
            RECORD_3_5_6
        with open("INPUT_RRTM", "w") as f:
            f.write(ret)
        return ret
    
    def create_inputfile_aerosols_solar(self, level=[0], aot=[0.19], num_aer=1, iaod=0, issa=0, ipha=0, aerpar=[0.13, 1.0, 0.0], ssa=[0.780], phase=[0.7]):
        
        RECORD_A1_1 = 3*" "+ "{:2d}".format(num_aer)
        RECORD_A2_1 = 3*" "+ "{:2d}".format(len(level))
        RECORD_A2_1+= 4*" "+ "{:1d}".format(iaod)
        RECORD_A2_1+= 4*" "+ "{:1d}".format(issa)
        RECORD_A2_1+= 4*" "+ "{:1d}".format(ipha)
        RECORD_A2_1+= "{:8.2f}".format(aerpar[0]) + "{:8.2f}".format(aerpar[1]) + "{:8.2f}".format(aerpar[2]) 
        RECORD_A2_1_1 = ""

        for i in enumerate(level):
            RECORD_A2_1_1 += 2*" " + "{:3d}".format(i[1]) + "{:7.4f}".format(aot[i[0]]) + "\n"
           
        RECORD_A2_2 = ""
        if iaod == 0:
            RECORD_A2_2 = "{:5.2f}".format(ssa[0])
        else:
            for i in range(14):
                RECORD_A2_2 += "{:5.2f}".format(ssa[i])
                
        RECORD_A2_3 = ""
        if ipha == 0:
            RECORD_A2_3 = "{:5.2f}".format(phase[0])
        else:
            for i in range(14):
                RECORD_A2_3 += "{:5.2f}".format(phase[i])
                
        ret = RECORD_A1_1 + "\n" + RECORD_A2_1 + "\n" + RECORD_A2_1_1 + RECORD_A2_2 + "\n" + RECORD_A2_3 + "\n"
        with open("IN_AER_RRTM", "w") as f:
            f.write(ret)
        return ret
        
    def create_inputfile_atm_terrestrial(self, cloud=0, tprof=[-1], pprof=[-1], hprof=[-1], zprof=[-1], semiss=[0.99], atm="4444444"):
        '''
        Create ASCII input for RRTMG containing informations about the atmosphere (terrestrial).
        For more informations refer to RRTMG
        '''

        temperature_prof = self.__cloud['temperature_prof'] if tprof[0] == -1 else tprof
        height_prof = self.__cloud['height_prof'] if zprof[0] == -1 else zprof
        humidity_prof = self.__cloud['humidity_prof'] if hprof[0] == -1 else hprof
        pressure_prof = self.__cloud['pressure_prof'] if pprof[0] == -1 else pprof
        RECORD_1_1 = "{:80s}".format("$ RRTM_LW runscript created on {}".format(dt.datetime.now()))
        IAER = 0
        IATM = 1
        IXSECT = 0
        NUMANGS = 0
        IOUT = 99
        IDRV = 0
        IMCA = 1 if cloud != 0 else 0
        #IMCA = 1
        ICLD = cloud
        RECORD_1_2  = 18 * " " + "{:2d}".format(IAER)
        RECORD_1_2 += 29 * " " + "{:1d}".format(IATM)
        RECORD_1_2 += 19 * " " + "{:1d}".format(IXSECT)
        RECORD_1_2 += 13 * " " + "{:2d}".format(NUMANGS)
        RECORD_1_2 += 2  * " " + "{:3d}".format(IOUT)
        RECORD_1_2 += 1  * " " + "{:1d}".format(IDRV)
        RECORD_1_2 += 1  * " " + "{:1d}".format(IMCA)
        RECORD_1_2 += 0  * " " + "{:1d}".format(ICLD)

        TBOUND = temperature_prof[0]
        IEMIS = 1 if len(semiss) == 1 else 2
        IREFLECT = 0
        SEMISS = semiss
        RECORD_1_4  = "{:>10.3f}".format(TBOUND) 
        RECORD_1_4 += 1  * " " + "{:1d}".format(IEMIS)
        RECORD_1_4 += 2  * " " + "{:1d}".format(IREFLECT)
        for element in SEMISS:
            RECORD_1_4 += "{:>5.3f}".format(element)

        MODEL = 0
        IBMAX = len(height_prof)
        NOPRINT = 0
        NMOL = 7
        IPUNCH = 1
        MUNITS = 1
        RE = 0
        RECORD_3_1  = "{:5d}".format(MODEL)
        RECORD_3_1 += 5 * " " + "{:5d}".format(IBMAX) 
        RECORD_3_1 += 5 * " " + "{:5d}".format(NOPRINT)
        RECORD_3_1 += "{:5d}".format(NMOL)
        RECORD_3_1 += "{:5d}".format(IPUNCH)
        RECORD_3_1 += 3 * " " + "{:2d}".format(MUNITS)
        RECORD_3_1 += "{:10.3f}".format(RE)


        HBOUND = height_prof[0]
        HTOA = height_prof[-1]
    
        RECORD_3_2 = "{:10.3f}{:10.3f}".format(HBOUND, HTOA)
        
        ZBND = ""
        for ii in range(len(height_prof)):
            ZBND += "{:10.3f}".format(height_prof[ii])
            if ii % 8 == 7:
                ZBND += "\n"
    
        RECORD_3_3B = ZBND
        
        IMMAX = len(height_prof)
        HMOD = ""

        ZM = height_prof
        PM = pressure_prof
        TM = temperature_prof
        JCHARP = "A"
        JCHART = "A"
        JCHAR = atm
        RECORD_3_5_6 = ""
        VOL = np.array([np.zeros(len(height_prof)) for ii in range(7)])
        VOL[0] = humidity_prof
        VOL[1] = self.__cloud['co2']
        VOL[2] = self.__cloud['o3']
        VOL[3] = self.__cloud['n2o']
        VOL[5] = self.__cloud['ch4']
        for loop in range(len(height_prof)):
            RECORD_3_5_6 += "{:10.3E}".format(ZM[loop])
            RECORD_3_5_6 += "{:10.3E}".format(PM[loop])
            RECORD_3_5_6 += "{:10.3f}".format(TM[loop])
            RECORD_3_5_6 += 5 * " " + "{:1s}".format(JCHARP) + "{:1s}".format(JCHART)
            RECORD_3_5_6 += 3 * " " + "{}".format(JCHAR)
            RECORD_3_5_6 += "\n"
            for molecules in range(7):
                RECORD_3_5_6 += "{:10.3E}".format(VOL[molecules][loop])
            RECORD_3_5_6 += "\n"
        
        RECORD_3_4 = "{:5d}".format(IMMAX) + "{:24s}".format(HMOD)
        ret = RECORD_1_1 + "\n" + RECORD_1_2 + "\n" + RECORD_1_4 + "\n" + \
            RECORD_3_1 + "\n" + RECORD_3_2 + "\n" + RECORD_3_3B + "\n" + RECORD_3_4 + "\n" + \
            RECORD_3_5_6
        with open("INPUT_RRTM", "w") as f:
            f.write(ret)

        return ret

    def create_inputfile_cloud(self, lay=[-1], cwp=[-1], clt=[-1], wpi=[-1], rliq=[-1], rice=[-1]):
        '''
        Create ASCII input for RRTMG containing informations about the cloud.
        For more informations refer to RRTMG
        '''
        INFLAG = 2
        ICEFLAG = 1
        LIQFLAG = 1
        RECORD_C1_1  = 4 * " " + "{:1d}".format(INFLAG)
        RECORD_C1_1 += 4 * " " + "{:1d}".format(ICEFLAG)
        RECORD_C1_1 += 4 * " " + "{:1d}".format(LIQFLAG)
        CLDFRAC = self.__cloud['clt'] if clt[0] == -1 else clt
        FRACICE = self.__cloud['wpi'] if wpi[0] == -1 else  wpi
        EFFSIZEICE = self.__cloud['rice'] if rice[0] == -1 else rice
        EFFSIZELIQ = self.__cloud['rliq'] if rliq[0] == -1 else rliq
        CWP = self.__cloud['cwp'] if cwp[0] == -1 else cwp
        LAY = self.__cloud['clevel'] if lay[0] == -1 else lay
        RECORD_C1_3 = ""
        ii = 0
            
        for i in enumerate(LAY):
            RECORD_C1_3 += " "
            RECORD_C1_3 += 1 * " " + "{:3d}".format(LAY[i[0]])
            RECORD_C1_3 += "{:>10.5f}".format(CLDFRAC[i[0]])
            RECORD_C1_3 += "{:>10.5f}".format(CWP[i[0]])
            RECORD_C1_3 += "{:>10.5f}".format(FRACICE[i[0]])
            RECORD_C1_3 += "{:>10.5f}".format(EFFSIZEICE[i[0]])
            RECORD_C1_3 += "{:>10.5f}".format(EFFSIZELIQ[i[0]])
            RECORD_C1_3 += "\n"
            ii += 1
        RECORD_C1_3 += "%"
            
        cld = RECORD_C1_1 + "\n" + RECORD_C1_3
        with open("IN_CLD_RRTM", "w") as f:
            f.write(cld)
        return cld

    def get_windows_terrestrial(self):
        return self.__wn_lw

    def get_windows_solar(self):
        return self.__wn_sw
    
    def run_RRTMG_terrestrial(self, clouds=True):
        subprocess.call(['{}'.format(self.__binary_lw)])
        for i in enumerate(self.__wn_lw):
            if clouds:
                self.__cloud['fluxes_lw_all'][i[0]] = read_results(len(self.__cloud['height_prof']), wavenumbers=[i[1][0], i[1][1]])
            else:
                self.__cloud['fluxes_lw_clear'][i[0]] = read_results(len(self.__cloud['height_prof']), wavenumbers=[i[1][0], i[1][1]])                     

    def run_RRTMG_solar(self, clouds=True):
        subprocess.call(['{}'.format(self.__binary_sw)])
        for i in enumerate(self.__wn_sw):
            if clouds:
                self.__cloud['fluxes_sw_all'][i[0]] = read_results(len(self.__cloud['height_prof']), wavenumbers=[i[1][0], i[1][1]])
            else:
                self.__cloud['fluxes_sw_clear'][i[0]] = read_results(len(self.__cloud['height_prof']), wavenumbers=[i[1][0], i[1][1]])           
        
    def get_fluxes_terrestrial(self, window):
        return {'all': self.__cloud['fluxes_lw_all'][window-1], 'clear': self.__cloud['fluxes_lw_clear'][window-1]}

    def get_fluxes_solar(self, window):
        return {'all': self.__cloud['fluxes_sw_all'][window-16], 'clear': self.__cloud['fluxes_sw_clear'][window-16]}

    def get_cparam(self):
        return {'LWP(gm-2)': np.array(self.__cloud['lwp']), \
                'IWP(gm-2)': np.array(self.__cloud['iwp']), \
                'rliq(um)': np.array(self.__cloud['rliq']), \
                'rice(um)': np.array(self.__cloud['rice']), \
                'clt': self.__cloud['clt'], \
                'red_chi_2': np.array(self.__cloud['red_chi_2'])}
            
    def get_atmosphere(self):
        return {'height': np.array(self.__cloud['height_prof']), \
                'pressure': np.array(self.__cloud['pressure_prof']), \
                'temperature': np.array(self.__cloud['temperature_prof']), \
                'humidity': np.array(self.__cloud['humidity_prof'])}

    def get_position(self):
        return {'Latitude': self.__cloud['latitude'], \
                'Longitude': self.__cloud['longitude'], \
                'SZA': self.__cloud['sza'], 'Time': self.__cloud['date']}
    
    def get_cloud(self):
        return self.__cloud

    
    def remove_rrtmg_files(self):
        files = ['tape6', 'TAPE6', 'TAPE7' 'INPUT_RRTM', 'IN_CLD_RRTM', 'OUTPUT_RRTM']
        for file_ in files:
            if os.path.exists(file_):
                os.remove(file_)

    def integrate_spectral_radiance(self, intervall):
        delta = 0.1
        radiance_f = scipy.interpolate.interp1d(self.__wavenumber, self.__spectrum, fill_value="extrapolate")
        radiance_integral = 0
        wn = intervall[0]
        while wn < intervall[1]:
            radiance_integral += radiance_f(wn)
            wn += delta
        
        return np.pi*1e-3*radiance_integral*delta

    def write_results(self, fname):
        '''
        Write results of RRTMG calculation to netCDF4-File
        '''
    
        with nc.Dataset(fname, "w") as outfile:
            outfile.createDimension("const", 1)
            outfile.createDimension("level", self.__cloud['height_prof'].size)
            outfile.createDimension('cgrid', self.__cloud['clevel'].size)
            
            height_out = outfile.createVariable("height", "f8", ("level", ))
            height_out.units = "km"
            height_out[:] = self.__cloud['height_prof'][:]
            
            pressure_out = outfile.createVariable("pressure", "f8", ("level", ))
            pressure_out.units = "hPa"
            pressure_out[:] = self.__cloud['pressure_prof'][:]
            
            temperature_out = outfile.createVariable("temperature", "f8", ("level", ))
            temperature_out.units = "K"
            temperature_out[:] = self.__cloud['temperature_prof'][:]
            
            humidity_out = outfile.createVariable("humidity", "f8", ("level", ))
            humidity_out.units = "%"
            humidity_out[:] = self.__cloud['humidity_prof'][:]
            
            lat_out = outfile.createVariable("latitude", "f8", ("const", ))
            lat_out.units = "DegN"
            lat_out[:] = self.__cloud['latitude']
    
            lon_out = outfile.createVariable("longitude", "f8", ("const", ))
            lon_out.units = "DegE"
            lon_out[:] = self.__cloud['longitude']
            
            sza_out = outfile.createVariable("solar_zenith_angle", "f8", ("const", ))
            sza_out.units = "Deg"
            sza_out[:] = self.__cloud['sza']
            
            sic_out = outfile.createVariable("sea_ice_concentration", "f8", ("const", ))
            sic_out.units = "1"
            sic_out[:] = self.__cloud['iceconc']
            
            albedo_dir_out = outfile.createVariable("sw_broadband_surface_albedo_direct_radiation", "f8", ("const", ))
            albedo_dir_out.units = "1"
            albedo_dir_out[:] = self.__cloud['albedo']
            
            albedo_diff_out = outfile.createVariable("sw_broadband_surface_albedo_diffuse_radiation", "f8", ("const", ))
            albedo_diff_out.units = "1"
            albedo_diff_out[:] = self.__cloud['albedo']
            
            cloud_out = outfile.createVariable("cloud_idx", "i4", ("cgrid", ))
            cloud_out[:] = self.__cloud['clevel'][:]
    
            cwp_out = outfile.createVariable("CWP", "f8", ("cgrid", ))
            cwp_out.units = "gm-2"
            cwp_out[:] = self.__cloud['cwp']
    
            rl_out = outfile.createVariable("rl", "f8", ("cgrid", ))
            rl_out.units = "um"
            rl_out[:] = self.__cloud['rliq']
    
            ri_out = outfile.createVariable("ri", "f8", ("cgrid", ))
            ri_out.units = "um"
            ri_out[:] = self.__cloud['rice']
    
            wpi_out = outfile.createVariable("WPi", "f8", ("cgrid", ))
            wpi_out.units = "1"
            wpi_out[:] = self.__cloud['wpi']
            
            clt_out = outfile.createVariable("cloud_fraction", "f8", ("cgrid", ))
            clt_out.units = "1"
            clt_out[:] = self.__cloud['clt']
            
            co2_out = outfile.createVariable("co2_profile", "f8", ("level", ))
            co2_out.units = "ppmv"
            co2_out[:] = self.__cloud['co2']
            
            n2o_out = outfile.createVariable("n2o_profile", "f8", ("level", ))
            n2o_out.units = "ppmv"
            n2o_out[:] = self.__cloud['n2o']
    
            ch4_out = outfile.createVariable("ch4_profile", "f8", ("level", ))
            ch4_out.units = "ppmv"
            ch4_out[:] = self.__cloud['ch4']
            
            diff_lwp_out = outfile.createVariable("diff_lwp", "f8", ("const", ))
            diff_lwp_out[:] = self.__cloud['diff_lwp']
            
            flag_lwc_out = outfile.createVariable('flag_lwc', 'f8', ('const', ))
            flag_lwc_out[:] = self.__cloud['flag_lwc']
            
            flag_iwc_out = outfile.createVariable('flag_iwc', 'f8', ('const', ))
            flag_iwc_out[:] = self.__cloud['flag_iwc']
            
            flag_reff_out = outfile.createVariable('flag_reff', 'f8', ('const', ))
            flag_reff_out[:] = self.__cloud['flag_reff']
            
            flag_reffice_out = outfile.createVariable('flag_reffice', 'f8', ('const', ))
            flag_reffice_out[:] = self.__cloud['flag_reffice']
            
            oob_liq = np.where((self.__cloud['rliq'] < 2.5) & \
                               (self.__cloud['wpi'] < 1.0))[0].size
            oob_liq+= np.where((self.__cloud['rliq'] >60.0) & \
                               (self.__cloud['wpi'] < 1.0))[0].size
            oob_ice = np.where((self.__cloud['rice'] < 13.0) & \
                               (self.__cloud['wpi'] > 0.0))[0].size
            oob_ice+= np.where((self.__cloud['rice'] > 131.0) & \
                               (self.__cloud['wpi'] > 0.0))[0].size
                    
            oob = outfile.createVariable("out_of_bounds", "i4", ("const", ))
            oob[:] = oob_liq + oob_ice
            
            keys_sw = self.__cloud['fluxes_sw_all'][0].keys()
            keys_lw = self.__cloud['fluxes_lw_all'][0].keys()
            clear_sw_out = [None for ii in range(len(keys_sw))]
            all_sw_out = [None for ii in range(len(keys_sw))]
            clear_sw_out_sum = [None for ii in range(len(keys_sw))]
            all_sw_out_sum = [None for ii in range(len(keys_sw))]
            for ii in range(len(keys_sw)):
                if keys_sw[ii] == "LEVEL" or keys_sw[ii] == "PRESSURE":
                    continue
                clear_sw_out[ii] = outfile.createVariable("clear_sw_{}".format(keys_sw[ii]), 'f8', ('level', ))
                all_sw_out[ii] = outfile.createVariable("all_sw_{}".format(keys_sw[ii]), 'f8', ('level', ))
                clear_sw_out_sum[ii] = outfile.createVariable("clear_sw_{}_integrated".format(keys_sw[ii]), 'f8', ('level', ))
                all_sw_out_sum[ii] = outfile.createVariable("all_sw_{}_integrated".format(keys_sw[ii]), 'f8', ('level', ))
                
                if keys_sw[ii] == "HEATING RATE":
                    all_sw_out[ii].units = "degree/day"
                    clear_sw_out[ii].units = "degree/day"
                    all_sw_out_sum[ii].units = "degree/day"
                    clear_sw_out_sum[ii].units = "degree/day"
                    
                else:
                    all_sw_out[ii].units = "Wm-2"
                    clear_sw_out[ii].units = "Wm-2"
                    all_sw_out_sum[ii].units = "Wm-2"
                    clear_sw_out_sum[ii].units = "Wm-2"
                    
                all_sw_out_int = np.zeros(self.__cloud['fluxes_sw_all'][0][keys_sw[0]].size)
                clear_sw_out_int = np.zeros(self.__cloud['fluxes_sw_all'][0][keys_sw[0]].size)

                for i in range(14):
                    all_sw_out_int += np.array(self.__cloud['fluxes_sw_all'][i][keys_sw[ii]])[::-1]
                    clear_sw_out_int += np.array(self.__cloud['fluxes_sw_clear'][i][keys_sw[ii]])[::-1]
                all_sw_out[ii][:] = all_sw_out_int#np.array(self.__fluxes_sw_all[0][keys_sw][ii])[::-1]
                clear_sw_out[ii][:] = clear_sw_out_int#np.array(self.__fluxes_sw_clear[0][keys_sw][ii])[::-1]
                all_sw_out_sum[ii][:] = np.array(self.__cloud['fluxes_sw_all'][-1][keys_sw[ii]])[::-1]
                clear_sw_out_sum[ii][:] = np.array(self.__cloud['fluxes_sw_clear'][-1][keys_sw[ii]])[::-1]
                
            clear_lw_out = [None for ii in range(len(keys_lw))]
            all_lw_out = [None for ii in range(len(keys_lw))]
            clear_lw_out_sum = [None for ii in range(len(keys_lw))]
            all_lw_out_sum = [None for ii in range(len(keys_lw))]
            for ii in range(len(keys_lw)):
                if keys_lw[ii] == "LEVEL" or keys_lw[ii] == "PRESSURE":
                    continue
                clear_lw_out[ii] = outfile.createVariable("clear_lw_{}".format(keys_lw[ii]), 'f8', ('level', ))
                all_lw_out[ii] = outfile.createVariable("all_lw_{}".format(keys_lw[ii]), 'f8', ('level', ))
                clear_lw_out_sum[ii] = outfile.createVariable("clear_lw_{}_integrated".format(keys_lw[ii]), 'f8', ('level', ))
                all_lw_out_sum[ii] = outfile.createVariable("all_lw_{}_integrated".format(keys_lw[ii]), 'f8', ('level', ))
                
                
                if keys_lw[ii] == "HEATING RATE":
                    all_lw_out[ii].units = "degree/day"
                    clear_lw_out[ii].units = "degree/day"
                    all_lw_out_sum[ii].units = "degree/day"
                    clear_lw_out_sum[ii].units = "degree/day"
                else:
                    all_lw_out[ii].units = "Wm-2"
                    clear_lw_out[ii].units = "Wm-2"
                    all_lw_out_sum[ii].units = "Wm-2"
                    clear_lw_out_sum[ii].units = "Wm-2"                    
                    
                all_lw_out_int = np.zeros(self.__cloud['fluxes_lw_all'][0][keys_lw[0]].size)
                clear_lw_out_int = np.zeros(self.__cloud['fluxes_lw_all'][0][keys_lw[0]].size)
                

                for i in range(16):
                    all_lw_out_int += np.array(self.__cloud['fluxes_lw_all'][i][keys_lw[ii]])[::-1]
                    clear_lw_out_int += np.array(self.__cloud['fluxes_lw_clear'][i][keys_lw[ii]])[::-1]
                all_lw_out[ii][:] = all_lw_out_int#np.array(self.__fluxes_sw_all[0][keys_sw][ii])[::-1]
                clear_lw_out[ii][:] = clear_lw_out_int#np.array(self.__fluxes_sw_clear[0][keys_sw][ii])[::-1]
                all_lw_out_sum[ii][:] = np.array(self.__cloud['fluxes_lw_all'][-1][keys_lw[ii]])[::-1]
                clear_lw_out_sum[ii][:] = np.array(self.__cloud['fluxes_lw_clear'][-1][keys_lw[ii]])[::-1]
                #all_lw_out[ii][:] = np.array(self.__fluxes_lw_all[0][keys_lw[ii]])[::-1]
                #clear_lw_out[ii][:] = np.array(self.__fluxes_lw_clear[0][keys_lw[ii]])[::-1]