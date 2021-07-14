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
                        [2600., 3250.]]
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
                        [820., 2600.]]
        self.__binary_lw = binary_lw
        self.__binary_sw = binary_sw
        [self.__liq_db, self.__ice_db] = mie.read_databases(mie_db[0], mie_db[1])

    def read_tcwret(self, fname, input_radiance=""):
        pattern = fname.split("/")[-1]
        self.__date = dt.datetime.strptime(pattern, "results_%Y%m%d%H%M%S.nc")
        time_dec = self.__date.hour + self.__date.minute/60.0 + self.__date.second/3600.0

        ## Read radiances from inputfiles of TCWret. This is done because the radiance output from TCWret is averaged
        ## for the microwindows. One can use the radiance output from TCWret if no path to the inputfiles is specified
        if input_radiance != "":
            with nc.Dataset(os.path.join(input_radiance, "PS.{:04d}{:02d}{:02d}.nc".format(self.__date.year, self.__date.month, self.__date.day))) as f:
                idx = np.argmin(np.abs(f.variables['time_dec'][:]-time_dec))
                self.__wavenumber = f.variables['wavenumber'][idx]
                self.__spectrum = f.variables['radiance'][idx]
        
        with nc.Dataset(fname, "r") as f:
            self.__latitude = np.float(f.variables['lat'][0])
            self.__longitude = np.float(f.variables['lon'][0])
            self.__height_prof = f.variables['z'][:]*1e-3
            self.__pressure_prof = f.variables['P'][:]
            self.__humidity_prof = f.variables['humidity'][:]
            self.__temperature_prof = f.variables['T'][:]
            self.__sza = np.float(f.variables['sza'][0])
            if input_radiance == "":
                self.__wavenumber = f.variables['wavenumber'][:]
                self.__spectrum = f.variables['lbldis radiance'][:]
        
            self.__red_chi_2 = f.variables['red_chi_2'][0]
            self.__xret = f.variables['x_ret'][:]
            self.__clevel = np.array(np.where(self.__height_prof == f.variables['clevel'][:]*1e-3)[0])
            self.__rliq = np.array([np.exp(f.variables['x_ret'][2])])
            self.__rice = np.array([np.exp(f.variables['x_ret'][3])])
            self.__lwp = np.array([mie.calc_lwp(self.__rliq, 0.0, self.__xret[0], 0.0)[0]])
            self.__iwp = np.array([mie.calc_iwp(self.__xret[1], 0.0, self.__rice, 0.0, self.__ice_db)[0]])
            self.__wpi = self.__iwp/(self.__lwp+self.__iwp)
            self.__clt = np.ones(self.__clevel.size)

    def plot_spectrum(self, ylim=[-1, -1], xlim=[-1, -1], fname=""):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(self.__wavenumber, self.__spectrum)
        ax.grid(True)
        if ylim != [-1, -1]: ax.set_ylim(ylim)
        if xlim != [-1, -1]: ax.set_xlim(xlim)
        if fname != "": plt.savefig(fname)
        plt.show()
            
    def plot_atmosphere(self, fname=""):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].set_title("Height coordinates")
        ax[1].set_title("Pressure coordinates")
        ax[0].plot(self.__humidity_prof, self.__height_prof, label="Humidity")
        ax[0].plot(self.__temperature_prof, self.__height_prof, label="Temperature")
        ax[0].hlines(self.__height_prof[self.__clevel], 0, 300, label="Cloud")
        ax[1].plot(self.__humidity_prof, self.__pressure_prof, label="Humidity")
        ax[1].plot(self.__temperature_prof, self.__pressure_prof, label="Temperature")
        ax[1].hlines(self.__pressure_prof[self.__clevel], 0, 300, label="Cloud")

        ax[1].set_ylim([1013, 0])
        for i in range(2):
            ax[i].grid(True)
            ax[i].legend()
        if fname != "": plt.savefig(fname)
        plt.show()

    def create_inputfile_atm_solar(self, cloud=0, tprof=[-1], pprof=[-1], hprof=[-1], zprof=[-1], albedo=[0.99], co2_mix=400):

        temperature_prof = self.__temperature_prof if tprof[0] == -1 else tprof
        height_prof = self.__height_prof if zprof[0] == -1 else zprof
        humidity_prof = self.__humidity_prof if hprof[0] == -1 else hprof
        pressure_prof = self.__pressure_prof if pprof[0] == -1 else pprof
        
        RECORD_1_1 = "{:80s}".format("$ RRTM_SW runscript created on {}".format(dt.datetime.now()))

        IAER = 0
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

        JULDAT = int((self.__date - dt.datetime(self.__date.year, 1, 1)).total_seconds()/(24*3600.0))
        SZA = self.__sza
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
        SEMISS = albedo
    
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
        CO2MX = co2_mix
        REF_LAT = 0
    
        RECORD_3_1  = "{:5d}".format(MODEL)
        RECORD_3_1 += 5 * " " + "{:5d}".format(IBMAX) 
        RECORD_3_1 += 5 * " " + "{:5d}".format(NOPRINT)
        RECORD_3_1 += "{:5d}".format(NMOL)
        RECORD_3_1 += "{:5d}".format(IPUNCH)
        RECORD_3_1 += 3 * " " + "{:2d}".format(MUNITS)
        RECORD_3_1 += "{:10.3f}".format(RE)
        RECORD_3_1 += 20 * " " + "{:10.3f}".format(CO2MX)
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
        JCHAR = "HA44444"
        RECORD_3_5_6 = ""
        VOL = np.array([np.zeros(len(height_prof)) for ii in range(7)])
        VOL[0] = humidity_prof
        VOL[1] = co2_mix * np.ones(humidity_prof.size)
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
        
    def create_inputfile_atm_terrestrial(self, cloud=0, tprof=[-1], pprof=[-1], hprof=[-1], zprof=[-1], semiss=[0.99], co2_mix=400):
        '''
        Create ASCII input for RRTMG containing informations about the atmosphere (terrestrial).
        For more informations refer to RRTMG
        '''

        temperature_prof = self.__temperature_prof if tprof[0] == -1 else tprof
        height_prof = self.__height_prof if zprof[0] == -1 else zprof
        humidity_prof = self.__humidity_prof if hprof[0] == -1 else hprof
        pressure_prof = self.__pressure_prof if pprof[0] == -1 else pprof
        
        RECORD_1_1 = "{:80s}".format("$ RRTM_LW runscript created on {}".format(dt.datetime.now()))
        IAER = 0
        IATM = 1
        IXSECT = 0
        NUMANGS = 0
        IOUT = 99
        IDRV = 0
        IMCA = 1 if cloud != 0 else 0
        IMCA = 1
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
        JCHAR = "HA44444"
        RECORD_3_5_6 = ""
        VOL = np.array([np.zeros(len(height_prof)) for ii in range(7)])
        VOL[0] = humidity_prof
        VOL[1] = co2_mix * np.ones(humidity_prof.size)
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
        CLDFRAC = self.__clt if clt[0] == -1 else clt
        FRACICE = self.__wpi if wpi[0] == -1 else  wpi
        EFFSIZEICE = self.__rice if rice[0] == -1 else rice
        EFFSIZELIQ = self.__rliq if rliq[0] == -1 else rliq
        CWP = self.__lwp+self.__iwp if cwp[0] == -1 else cwp
        LAY = self.__clevel if lay[0] == -1 else lay
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
    
    def run_RRTMG_terrestrial(self):
        subprocess.call(['{}'.format(self.__binary_lw)])
        self.__fluxes_lw = [None for i in range(16)]
        for i in enumerate(self.__wn_lw):
            self.__fluxes_lw[i[0]] = read_results(len(self.__height_prof), wavenumbers=[i[1][0], i[1][1]])

    def run_RRTMG_solar(self):
        subprocess.call(['{}'.format(self.__binary_sw)])
        self.__fluxes_sw = [None for i in range(14)]
        for i in enumerate(self.__wn_sw):
            self.__fluxes_sw[i[0]] = read_results(len(self.__height_prof), wavenumbers=[i[1][0], i[1][1]])
        
    def get_fluxes_terrestrial(self, window):
        return self.__fluxes_lw[window-1]

    def get_fluxes_solar(self, window):
        return self.__fluxes_sw[window-16]

    def get_cparam(self):
        return {'LWP(gm-2)': np.array(self.__lwp), \
                'IWP(gm-2)': np.array(self.__iwp), \
                'rliq(um)': np.array(self.__rliq), \
                'rice(um)': np.array(self.__rice), \
                'clt': self.__clt, \
                'red_chi_2': np.array(self.__red_chi_2)}

    def get_position(self):
        return {'Latitude': self.__latitude, \
                'Longitude': self.__longitude, \
                'SZA': self.__sza}
    
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
    
    def run(self):
        cparam = self.get_cparam()
        position = self.get_position()
        
        in_cld_rrtm = self.create_inputfile_cloud()
        input_rrtm = self.create_inputfile_atm_terrestrial(cloud=2, co2_mix=400)
        self.run_RRTMG_terrestrial()
        self.remove_rrtmg_files()
        in_cld_rrtm = self.create_inputfile_cloud()
        input_rrtm = self.create_inputfile_atm_solar(cloud=2, co2_mix=400)
        self.run_RRTMG_solar()
        self.remove_rrtmg_files()
        flux_all = []
        flux_integrate = []
        for win in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
            flux_all.append(self.get_fluxes_terrestrial(win)['DOWNWARD FLUX'].iloc[-1])
            flux_integrate.append(self.integrate_spectral_radiance(model.get_windows_terrestrial()[win-1]))
            
        for win in [16,17,18,19,20,21,22,23,24,25,26,27,28,29]:
            flux_all.append(self.get_fluxes_solar(win)['DOWNWARD FLUX'].iloc[-1])
            
        input_rrtm = self.create_inputfile_atm_terrestrial(cloud=0, co2_mix=400)
        self.run_RRTMG_terrestrial()
        self.remove_rrtmg_files()
        in_cld_rrtm = self.create_inputfile_cloud()
        input_rrtm = self.create_inputfile_atm_solar(cloud=0, co2_mix=400)
        self.run_RRTMG_solar()
        self.remove_rrtmg_files()

        flux_clear = []
        for win in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
            flux_clear.append(self.get_fluxes_terrestrial(win)['DOWNWARD FLUX'].iloc[-1])
            
        for win in [16,17,18,19,20,21,22,23,24,25,26,27,28,29]:
            flux_clear.append(self.get_fluxes_solar(win)['DOWNWARD FLUX'].iloc[-1])
            
        return cparam, position, flux_all, flux_integrate, flux_clear, self.__date
if __name__ == '__main__':
    path_retrievals = sys.argv[1]
    path_input = sys.argv[2]
    ssp_ice = sys.argv[3]
    ssp_wat = sys.argv[4]
    binary_lw = sys.argv[5]
    binary_sw = sys.argv[6]
    outpath = sys.argv[7]
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    os.chdir(outpath)
    files = sorted(os.listdir(path_retrievals))
    
    model = RRTMG(binary_lw, binary_sw, [ssp_wat, ssp_ice])
    out = {'Date': [], 'Latitude': [], 'Longitude': [], 'SZA': [], 'LWP': [], 'IWP': [], 'rliq': [], 'rice': [], 'red_chi_2': []}
    for i in range(16):
        out.update({'FTIR_integrate_{:02d}'.format(i): []})
    for i in range(29):
        out.update({'RRTMG_All_{:02d}'.format(i): []})
        out.update({'RRTMG_Clear_{:02d}'.format(i): []})
    for spec in files:
        rrtmg_all = []
        rrtmg_clear = []
        ftir_all = []
        model.read_tcwret(os.path.join(path_retrievals, spec), path_input)
        try:
            cparam, position, flux_all, flux_integrate, flux_clear, date = model.run()
            out['Date'].append(date)
            out['Latitude'].append(position['Latitude'])
            out['Longitude'].append(position['Longitude'])
            out['SZA'].append(position['SZA'])
            out['LWP'].append(np.float(cparam['LWP(gm-2)'].flatten()))
            out['IWP'].append(np.float(cparam['IWP(gm-2)'].flatten()))
            out['rliq'].append(np.float(cparam['rliq(um)'].flatten()))
            out['rice'].append(np.float(cparam['rice(um)'].flatten()))
            out['red_chi_2'].append(np.float(cparam['red_chi_2'].flatten()))
            for i in range(29):
                out['RRTMG_All_{:02d}'.format(i)].append(flux_all[i])
                out['RRTMG_Clear_{:02d}'.format(i)].append(flux_clear[i])

            for i in range(16):
                out['FTIR_integrate_{:02d}'.format(i)].append(flux_integrate[i])
        except UnboundLocalError:
            continue
    
    pd.DataFrame(out).to_csv("out.csv", index=False)
    