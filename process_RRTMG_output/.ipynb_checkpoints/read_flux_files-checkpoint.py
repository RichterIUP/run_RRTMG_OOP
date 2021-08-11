import sys
import netCDF4 as nc
import pandas as pd
import numpy as np
import os
import datetime as dt
from scipy.interpolate import interp1d

SFC = 0
TOA = -1

def interpolate_ERA5(data, lat_int, lon_int, time_int):
    '''
    Find nearest value of given key in ERA5 single level file
    '''
    tim = (time_int - dt.datetime(1900, 1, 1)).total_seconds()/3600

    latitude = data[1]#f.variables['latitude'][:]
    longitude = data[2]#f.variables['longitude'][:]
    time = data[0]#f.variables['time'][:]
    value = data[-1]#f.variables[key][:]
    ret_val = np.array([])
    lat_min = np.abs(latitude - lat_int)
    lon_min = np.abs(longitude - lon_int)
    time_min = np.abs(time - tim)
    argmin_lat = np.argmin(lat_min)
    argmin_lon = np.argmin(lon_min)
    argmin_time = np.argmin(time_min)
    ret_val = value[argmin_time, argmin_lat, argmin_lon]
    return ret_val

def open_era5(fname, key):
    with nc.Dataset(fname) as f:
        time = f.variables['time'][:]
        latitude = f.variables['latitude'][:]
        longitude = f.variables['longitude'][:]
        cc = f.variables[key][:]
    return [time, latitude, longitude, cc]

def read_flux_files(path, fname_out, idx_date, idx_time):
    files = sorted(os.listdir(path))
    
    #era5_sp = open_era5("/mnt/beegfs/user/phi.richter/DATA_PHD/ERA5/sfc_sl_pressure.nc", "msl")
    
    out = {'Albedo(1)': [], \
           'SIC(1)': [], \
           'cwp(gm-2)': [], \
           'toa_down_flux_all_lw(Wm-2)': [], \
           'toa_down_flux_clear_lw(Wm-2)': [], \
           'toa_down_flux_all_sw(Wm-2)': [], \
           'toa_up_flux_all_lw(Wm-2)': [], \
           'toa_up_flux_clear_lw(Wm-2)': [], \
           'toa_up_flux_all_sw(Wm-2)': [], \
           'toa_down_flux_clear_sw(Wm-2)': [], \
           'toa_up_flux_clear_sw(Wm-2)': [], \
           'toa_direct_flux_all_sw(Wm-2)': [], \
           'toa_direct_flux_clear_sw(Wm-2)': [], \
           'sfc_down_flux_all_lw(Wm-2)': [], \
           'sfc_down_flux_clear_lw(Wm-2)': [], \
           'sfc_down_flux_all_sw(Wm-2)': [], \
           'sfc_up_flux_all_lw(Wm-2)': [], \
           'sfc_up_flux_clear_lw(Wm-2)': [], \
           'sfc_up_flux_all_sw(Wm-2)': [], \
           'sfc_down_flux_clear_sw(Wm-2)': [], \
           'sfc_up_flux_clear_sw(Wm-2)': [], \
           'sfc_direct_flux_all_sw(Wm-2)': [], \
           'sfc_direct_flux_clear_sw(Wm-2)': [], \
           'sfc_heating_rate_all_sw(K/day)': [], \
           'sfc_heating_rate_clear_sw(K/day)': [], \
           'sfc_heating_rate_all_lw(K/day)': [], \
           'sfc_heating_rate_clear_lw(K/day)': [], \
           'sfc_cre_lw(Wm-2)': [], \
           'sfc_cre_sw(Wm-2)': [], \
           'sfc_cre(Wm-2)': [], \
           'sfc_heating_rate_lw(K/day)': [], \
           'sfc_heating_rate_sw(K/day)': [], \
           'sfc_heating_rate(K/day)': [], \
           'latitude': [], \
           'longitude': [], \
           'ri(um)': [], \
           'rl(um)': [], \
           't_surf(K)': [], \
           't_cld(K)': [], \
           'sza': [], \
           'time': [], \
           'wpi(1)': [], \
           'dCWP(gm-2)': [], \
           'diff_lwp(gm-2)': [], \
           'flag_lwc': [], \
           'flag_iwc': [], \
           'flag_reff': [], \
           'flag_rice': [], \
           'dWPi(1)': [], \
           'drl(um)': [], \
           'dri(um)': [], \
           'cbh(m)': [], 'clt(1)': [], 'lwp(gm-2)': [], 'iwp(gm-2)': []}
    
    for file_2 in files:
        file_ = os.path.join(path, file_2)
        if not ".nc" in file_:
            continue
        try:
            #if True:
            with nc.Dataset(os.path.join(os.path.join(path, file_2), file_)) as f:
                #print(file_)
               
                file_date = file_.split("/")[-1]
                year = int(file_date.split("_")[idx_date][:4])
                month = int(file_date.split("_")[idx_date][4:6])
                day = int(file_date.split("_")[idx_date][6:])
                hour = int(file_date.split("_")[idx_time][:2])
                minute = int(file_date.split("_")[idx_time][2:4])
                second = 0
                out['latitude'].append(np.float(f.variables['latitude'][:]))
                out['longitude'].append(np.float(f.variables['longitude'][:]))
                out['sza'].append(np.float(f.variables['solar_zenith_angle'][:]))
                out['time'].append(dt.datetime(year, month, day, hour, minute, second))
                
                out['diff_lwp(gm-2)'].append(np.float(f.variables['diff_lwp'][0]))
                out['flag_lwc'].append(np.float(f.variables['flag_lwc'][0]))
                out['flag_iwc'].append(np.float(f.variables['flag_lwc'][0]))
                out['flag_reff'].append(np.float(f.variables['flag_reff'][0]))
                out['flag_rice'].append(np.float(f.variables['flag_reffice'][0]))
                out['Albedo(1)'].append(np.float(f.variables['sw_broadband_surface_albedo_direct_radiation'][0]))
                out['SIC(1)'].append(np.float(f.variables['sea_ice_concentration'][0]))
                cwp = np.sum(f.variables['CWP'][:])
                cloud = f.variables['cloud_idx'][:]#np.where(f.variables['CWP'][:] > 0.0)
                wpi = np.mean(f.variables['WPi'][:])
                #print(wpi)
                rliq = np.mean(f.variableſ['rl'][:])
                rice = np.mean(f.variables['ri'][:])
                if rliq.size == 0:
                    rliq = -1
                if rice.size == 0:
                    rice = -1
                out['lwp(gm-2)'].append(np.sum(f.variables['CWP'][:]*(1-f.variables['WPi'][:])))
                out['iwp(gm-2)'].append(np.sum(f.variables['CWP'][:]*(f.variables['WPi'][:])))
                out['cwp(gm-2)'].append(cwp)
                out['wpi(1)'].append(wpi)
                if len(cloud) != 0:
                    out['cbh(m)'].append(np.float(f.variables['height'][cloud[0]]))
                    out['rl(um)'].append(rliq)
                    out['ri(um)'].append(rice)
                else:
                    cloud = np.array([0])
                    out['rl(um)'].append(-1)
                    out['ri(um)'].append(-1)
                    out['cbh(m)'].append(0)
                    
                sp = -1.0#interpolate_ERA5(era5_sp, out['latitude'][-1], out['longitude'][-1], out['time'][-1])*1e-2
                if sp > 0.0:
                    pressure = f.variables['pressure'][:]                   
                    sfc_down_flux_all_lw_f = interp1d(pressure, f.variables['all_lw_DOWNWARD FLUX'][:], fill_value="extrapolate")
                    sfc_down_flux_clear_lw_f = interp1d(pressure, f.variables['clear_lw_DOWNWARD FLUX'][:], fill_value="extrapolate")
                    sfc_up_flux_all_lw_f = interp1d(pressure, f.variables['all_lw_UPWARD FLUX'][:], fill_value="extrapolate")
                    sfc_up_flux_clear_lw_f = interp1d(pressure, f.variables['clear_lw_UPWARD FLUX'][:], fill_value="extrapolate")
                    sfc_down_flux_all_sw_f = interp1d(pressure, f.variables['all_sw_DOWNWARD FLUX'][:], fill_value="extrapolate")
                    sfc_down_flux_clear_sw_f = interp1d(pressure, f.variables['clear_sw_DOWNWARD FLUX'][:], fill_value="extrapolate")
                    sfc_up_flux_all_sw_f = interp1d(pressure, f.variables['all_sw_UPWARD FLUX'][:], fill_value="extrapolate")
                    sfc_up_flux_clear_sw_f = interp1d(pressure, f.variables['clear_sw_UPWARD FLUX'][:], fill_value="extrapolate")
                    sfc_direct_flux_all_sw_f = interp1d(pressure, f.variables['all_sw_DIRDOWN FLUX'][:], fill_value="extrapolate")
                    sfc_direct_flux_clear_sw_f = interp1d(pressure, f.variables['clear_sw_DIRDOWN FLUX'][:], fill_value="extrapolate")

                    sfc_heating_rate_all_lw_f = interp1d(pressure, f.variables['all_lw_HEATING RATE'][:], fill_value="extrapolate")
                    sfc_heating_rate_clear_lw_f = interp1d(pressure, f.variables['clear_lw_HEATING RATE'][:], fill_value="extrapolate")
                    sfc_heating_rate_all_sw_f = interp1d(pressure, f.variables['all_sw_HEATING RATE'][:], fill_value="extrapolate")
                    sfc_heating_rate_clear_sw_f = interp1d(pressure, f.variables['clear_sw_HEATING RATE'][:], fill_value="extrapolate")
                    
                    out['sfc_down_flux_all_lw(Wm-2)'].append(np.float(sfc_down_flux_all_lw_f(sp)))
                    out['sfc_down_flux_clear_lw(Wm-2)'].append(np.float(sfc_down_flux_clear_lw_f(sp)))
                    out['sfc_up_flux_all_lw(Wm-2)'].append(np.float(sfc_up_flux_all_lw_f(sp)))
                    out['sfc_up_flux_clear_lw(Wm-2)'].append(np.float(sfc_up_flux_clear_lw_f(sp)))
                    out['sfc_down_flux_all_sw(Wm-2)'].append(np.float(sfc_down_flux_all_sw_f(sp)))
                    out['sfc_down_flux_clear_sw(Wm-2)'].append(np.float(sfc_down_flux_clear_sw_f(sp)))
                    out['sfc_up_flux_all_sw(Wm-2)'].append(np.float(sfc_up_flux_all_sw_f(sp)))
                    out['sfc_up_flux_clear_sw(Wm-2)'].append(np.float(sfc_up_flux_clear_sw_f(sp)))
                    out['sfc_direct_flux_all_sw(Wm-2)'].append(np.float(sfc_direct_flux_all_sw_f(sp)))
                    out['sfc_direct_flux_clear_sw(Wm-2)'].append(np.float(sfc_direct_flux_clear_sw_f(sp)))
                
                    out['sfc_heating_rate_all_lw(K/day)'].append(np.float(sfc_heating_rate_all_lw_f(sp)))
                    out['sfc_heating_rate_clear_lw(K/day)'].append(np.float(sfc_heating_rate_clear_lw_f(sp)))
                    out['sfc_heating_rate_all_sw(K/day)'].append(np.float(sfc_heating_rate_all_sw_f(sp)))
                    out['sfc_heating_rate_clear_sw(K/day)'].append(np.float(sfc_heating_rate_clear_sw_f(sp)))
                else:
                    out['sfc_down_flux_all_lw(Wm-2)'].append(np.float(f.variables['all_lw_DOWNWARD FLUX_integrated'][SFC]))
                    out['sfc_down_flux_clear_lw(Wm-2)'].append(np.float(f.variables['clear_lw_DOWNWARD FLUX_integrated'][SFC]))
                    out['sfc_up_flux_all_lw(Wm-2)'].append(np.float(f.variables['all_lw_UPWARD FLUX_integrated'][SFC]))
                    out['sfc_up_flux_clear_lw(Wm-2)'].append(np.float(f.variables['clear_lw_UPWARD FLUX_integrated'][SFC]))
                    out['sfc_down_flux_all_sw(Wm-2)'].append(np.float(f.variables['all_sw_DOWNWARD FLUX_integrated'][SFC]))
                    out['sfc_down_flux_clear_sw(Wm-2)'].append(np.float(f.variables['clear_sw_DOWNWARD FLUX_integrated'][SFC]))
                    out['sfc_up_flux_all_sw(Wm-2)'].append(np.float(f.variables['all_sw_UPWARD FLUX_integrated'][SFC]))
                    out['sfc_up_flux_clear_sw(Wm-2)'].append(np.float(f.variables['clear_sw_UPWARD FLUX_integrated'][SFC]))
                    out['sfc_direct_flux_all_sw(Wm-2)'].append(np.float(f.variables['all_sw_DIRDOWN FLUX_integrated'][SFC]))
                    out['sfc_direct_flux_clear_sw(Wm-2)'].append(np.float(f.variables['clear_sw_DIRDOWN FLUX_integrated'][SFC]))
                
                    out['sfc_heating_rate_all_lw(K/day)'].append(np.float(f.variables['all_lw_HEATING RATE_integrated'][SFC]))
                    out['sfc_heating_rate_clear_lw(K/day)'].append(np.float(f.variables['clear_lw_HEATING RATE_integrated'][SFC]))
                    out['sfc_heating_rate_all_sw(K/day)'].append(np.float(f.variables['all_sw_HEATING RATE_integrated'][SFC]))
                    out['sfc_heating_rate_clear_sw(K/day)'].append(np.float(f.variables['clear_sw_HEATING RATE_integrated'][SFC]))
                
                out['toa_down_flux_all_lw(Wm-2)'].append(np.float(f.variables['all_lw_DOWNWARD FLUX_integrated'][TOA]))
                out['toa_down_flux_clear_lw(Wm-2)'].append(np.float(f.variables['clear_lw_DOWNWARD FLUX_integrated'][TOA]))
                out['toa_up_flux_all_lw(Wm-2)'].append(np.float(f.variables['all_lw_UPWARD FLUX_integrated'][TOA]))
                out['toa_up_flux_clear_lw(Wm-2)'].append(np.float(f.variables['clear_lw_UPWARD FLUX_integrated'][TOA]))
                out['toa_down_flux_all_sw(Wm-2)'].append(np.float(f.variables['all_sw_DOWNWARD FLUX_integrated'][TOA]))
                out['toa_down_flux_clear_sw(Wm-2)'].append(np.float(f.variables['clear_sw_DOWNWARD FLUX_integrated'][TOA]))
                out['toa_up_flux_all_sw(Wm-2)'].append(np.float(f.variables['all_sw_UPWARD FLUX_integrated'][TOA]))
                out['toa_up_flux_clear_sw(Wm-2)'].append(np.float(f.variables['clear_sw_UPWARD FLUX_integrated'][TOA]))
                out['toa_direct_flux_all_sw(Wm-2)'].append(np.float(f.variables['all_sw_DIRDOWN FLUX_integrated'][TOA]))
                out['toa_direct_flux_clear_sw(Wm-2)'].append(np.float(f.variables['clear_sw_DIRDOWN FLUX_integrated'][TOA]))
                
                out['t_surf(K)'].append(np.float(f.variables['temperature'][0]))
                out['t_cld(K)'].append(np.mean(f.variables['temperature'][cloud]))
                
                out['sfc_cre_lw(Wm-2)'].append(out['sfc_down_flux_all_lw(Wm-2)'][-1] - out['sfc_down_flux_clear_lw(Wm-2)'][-1] - \
                                               (out['sfc_up_flux_all_lw(Wm-2)'][-1] - out['sfc_up_flux_clear_lw(Wm-2)'][-1]))
                out['sfc_cre_sw(Wm-2)'].append(out['sfc_down_flux_all_sw(Wm-2)'][-1] - out['sfc_down_flux_clear_sw(Wm-2)'][-1] - \
                                               (out['sfc_up_flux_all_sw(Wm-2)'][-1] - out['sfc_up_flux_clear_sw(Wm-2)'][-1]))
                out['sfc_cre(Wm-2)'].append(out['sfc_cre_sw(Wm-2)'][-1]+out['sfc_cre_lw(Wm-2)'][-1])
                
                out['sfc_heating_rate_lw(K/day)'].append(out['sfc_heating_rate_all_lw(K/day)'][-1] - out['sfc_heating_rate_clear_lw(K/day)'][-1])
                out['sfc_heating_rate_sw(K/day)'].append(out['sfc_heating_rate_all_sw(K/day)'][-1] - out['sfc_heating_rate_clear_sw(K/day)'][-1])
                out['sfc_heating_rate(K/day)'].append(out['sfc_heating_rate_sw(K/day)'][-1]+out['sfc_heating_rate_lw(K/day)'][-1])
    
                #out['d_cwp_lw_DOWNWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_cwp_lw_DOWNWARD FLUX'][cloud]))
                #out['d_cwp_sw_DOWNWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_cwp_sw_DOWNWARD FLUX'][cloud]))
                #out['d_rl_lw_DOWNWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_rl_lw_DOWNWARD FLUX'][cloud]))
                #out['d_rl_sw_DOWNWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_rl_sw_DOWNWARD FLUX'][cloud]))
                #out['d_ri_lw_DOWNWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_ri_lw_DOWNWARD FLUX'][cloud]))
                #out['d_ri_sw_DOWNWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_ri_sw_DOWNWARD FLUX'][cloud]))
                #out['d_wpi_lw_DOWNWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_wpi_lw_DOWNWARD FLUX'][cloud]))
                #out['d_wpi_sw_DOWNWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_wpi_sw_DOWNWARD FLUX'][cloud]))
                    
                #out['d_cwp_lw_UPWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_cwp_lw_UPWARD FLUX'][cloud]))
                #out['d_cwp_sw_UPWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_cwp_sw_UPWARD FLUX'][cloud]))
                #out['d_rl_lw_UPWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_rl_lw_UPWARD FLUX'][cloud]))
                #out['d_rl_sw_UPWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_rl_sw_UPWARD FLUX'][cloud]))
                #out['d_ri_lw_UPWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_ri_lw_UPWARD FLUX'][cloud]))
                #out['d_ri_sw_UPWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_ri_sw_UPWARD FLUX'][cloud]))
                #out['d_wpi_lw_UPWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_wpi_lw_UPWARD FLUX'][cloud]))
                #out['d_wpi_sw_UPWARD FLUX'].append(\
                #         np.mean(f.variables['difference_quotient_wpi_sw_UPWARD FLUX'][cloud]))
    
                #delta_cwp_up_flux_all_lw = np.mean(f.variables['difference_quotient_cwp_lw_UPWARD FLUX'][cloud])
                #delta_cwp_up_flux_all_sw = np.mean(f.variables['difference_quotient_cwp_sw_UPWARD FLUX'][cloud])
                #delta_rl_up_flux_all_lw = np.mean(f.variables['difference_quotient_rl_lw_UPWARD FLUX'][cloud])
                #delta_rl_up_flux_all_sw = np.mean(f.variables['difference_quotient_rl_sw_UPWARD FLUX'][cloud])
                #delta_ri_up_flux_all_lw = np.mean(f.variables['difference_quotient_ri_lw_UPWARD FLUX'][cloud])
                #delta_ri_up_flux_all_sw = np.mean(f.variables['difference_quotient_ri_sw_UPWARD FLUX'][cloud])
                #delta_wpi_up_flux_all_lw = np.mean(f.variables['difference_quotient_wpi_lw_UPWARD FLUX'][cloud])
                #delta_wpi_up_flux_all_sw = np.mean(f.variables['difference_quotient_wpi_sw_UPWARD FLUX'][cloud])
    
                dcwp = 0.0#np.sum(f.variables['delta_CWP'][:])
                dwpi = 0.0#np.mean(f.variables['delta_WPi'][:])
                drliq = 0.0#np.mean(f.variables['delta_rl'][:])
                drice = 0.0#np.mean(f.variables['delta_ri'][:])
                if np.isnan(dcwp):  dcwp  = 0.0
                if np.isnan(dwpi):  dwpi  = 0.0
                if np.isnan(drliq): drliq = 0.0
                if np.isnan(drice): drice = 0.0
                out['dCWP(gm-2)'].append(dcwp)
                out['dWPi(1)'].append(dwpi)
                out['drl(um)'].append(drliq)
                out['dri(um)'].append(drice)
                try:
                    out['clt(1)'].append(np.mean(f.variables['cloud_fraction'][:]))
                except KeyError:
                    out['clt(1)'].append(1)
                except ValueError:
                    out['clt(1)'].append(-1)
                #print(dcwp, dwpi, drliq, drice)
                #if np.isinf(dcwp) or np.isinf(dwpi) or np.isinf(drliq) or np.isinf(drice) or \
                #    np.isnan(dcwp) or np.isnan(dwpi) or np.isnan(drliq) or np.isnan(drice) or \
                #    np.isnan(delta_cwp_down_flux_all_lw * dcwp + delta_rl_down_flux_all_lw * drliq + delta_ri_down_flux_all_lw * drice + delta_wpi_down_flux_all_lw * dwpi) or np.ma.is_masked(dcwp) or np.ma.is_masked(dwpi) or np.ma.is_masked(drliq) or np.ma.is_masked(drice):
                #    #print("Cont")
                #    continue
                #    #pass
     
            #break
        except Exception:
            print(file_)
            pass
    
    pd.DataFrame(out).to_csv(fname_out, index=False)
    #exit(-1)
    #print(out, np.float(out['diff_lwp(gm-2)']), np.float(out['net_cre_lw(Wm-2)']), np.float(out['net_cre_sw(Wm-2)']))
    #out.to_csv(fname_out, index=False)

if __name__ == '__main__':
    #read_flux_files("/mnt/beegfs/user/phi.richter/DATA_PHD/RRTMG/OUTPUT/RRTMG_CNET_nomod_w_diffquot", "RRTMG_CNET_nomod_w_diffquot_w_cre.csv", 3, 4)
    #read_flux_files("/home/philipp/PHD/DATA_PHD/RRTMG/OUTPUT/RRTMG_ERA5_nomod_correct_clt", "RRTMG_ERA5_nomod_correct_clt.csv", 2, 3)
    #read_flux_files("/home/philipp/PHD/DATA_PHD/RRTMG/OUTPUT/RRTMG_ERA5_scale", "RRTMG_ERA5_scale.csv", 2, 3)
    #read_flux_files("/home/philipp/PHD/DATA_PHD/RRTMG/OUTPUT/RRTMG_ERA5_pwp", "RRTMG_ERA5_pwp.csv", 2, 3)
    #read_flux_files("/home/philipp/PHD/DATA_PHD/RRTMG/OUTPUT/RRTMG_ERA5_mwp", "RRTMG_ERA5_mwp.csv", 2, 3)

    #read_flux_files("/home/phi.richter/SOFTWARE_PHD/run_RRTMG/RRTMG_ERA5_virtual_temperature_correct_clt_in_cld", "RRTMG_ERA5_virtual_temperature_correct_clt_in_cld.csv", 2, 3)
    #read_flux_files("/home/phi.richter/SOFTWARE_PHD/run_RRTMG/RRTMG_Cloudnet", "RRTMG_CNET.csv", 3, 4)
    read_flux_files(sys.argv[1], sys.argv[2], 3, 4)
