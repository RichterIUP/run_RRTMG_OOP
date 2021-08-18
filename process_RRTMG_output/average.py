#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd

"""
Created on Fri Dec  4 12:32:27 2020

@author: philipp
"""

def average_datasets_mult(data_average, key, time_init, time_end, delta, time_key='date time', keys_std=["cwp(gm-2)"]):
    # Find corresponding entries and average
    # Criterion: +- delta seconds. Averaging all datapoints inbetween 
    idx = []
    date = []
    time_now = time_init
    while time_now < time_end:
        idx_0 = np.where(data_average[time_key] >= time_now-pd.Timedelta(delta, "m"))
        idx_1 = np.where(data_average[time_key] < time_now)
        intersect = np.intersect1d(idx_0, idx_1)
        if len(intersect) > 0:
            idx.append(intersect)
            date.append(time_now)
        time_now += pd.Timedelta(delta, "m")
        
    mean_key  = [[] for ii in range(len(key))]
    std_key = [[] for ii in range(len(keys_std))]
    mean_date = []
    num = []
    for ii in range(len(idx)):
        if date[ii] not in mean_date:
            num.append(idx[ii].size)
            mean_date.append(date[ii])
            for kk in range(len(key)):
                try:
                    x = np.mean(np.array(data_average[key[kk]].iloc[idx[ii]], dtype=np.float64))
                    if np.isnan(x):
                        mean_key[kk].append(-1)
                    else:
                        mean_key[kk].append(x)
                    if key[kk] in keys_std:
                        jj = keys_std.index(key[kk])
                        x = np.std(np.array(data_average[keys_std[jj]].iloc[idx[ii]], dtype=np.float64))
                        if np.isnan(x):
                            std_key[jj].append(-1)
                        else:
                            std_key[jj].append(x)
                except ValueError:
                    print(data_average[key[kk]].iloc[idx[ii]])
    out_dict = {time_key : mean_date, 'datapoints' : num}
    for kk in range(len(key)):
        out_dict.update({key[kk] : mean_key[kk]})
    for kk in range(len(keys_std)):
        out_dict.update({"std_"+keys_std[kk] : std_key[kk]})
    mean_corresponding = pd.DataFrame(out_dict) 
                
    return mean_corresponding

def average(fname_in, delta_t, fname_out, filter_invalid=True):
    
    results = pd.read_csv(fname_in, parse_dates=['time'])

    ## Discard invalid results
    idx_valid = np.array([])
    
    for ii in range(len(results)):
        ## Ignore results, where LW downward flux is 0.0. 
        ## This indicates a problem with the RRTMG calculation
        if results['sfc_down_flux_all_lw(Wm-2)'].iloc[ii] == 0.0:
            continue
        ## Ignore results with invalid flags or wrong inconsistent LWP
        elif filter_invalid and ((np.abs(results['diff_lwp(gm-2)'].iloc[ii]) > 1.0 and \
             np.abs(results['diff_lwp(gm-2)'].iloc[ii]) < 2000.0) or results['flag_lwc'].iloc[ii] > 0 or results['flag_iwc'].iloc[ii] > 0):
            continue
        else:
            idx_valid = np.concatenate((idx_valid, [ii]))
        
    results  = results.iloc[idx_valid]

    ## Find column containing dates
    with open(fname_in, "r") as f:
        keys = f.readline().split(",")

    column_date = np.where(np.array(keys) == "time")[0][0]
    keys.pop(column_date)
    keys = [key.rstrip() for key in keys]

    if delta_t > 0.:
        results = average_datasets_mult(results, keys, \
                                      time_init=pd.Timestamp("2017-05-30T00:00:00"), \
                                      time_end=pd.Timestamp("2017-07-19T00:00:00"), delta=delta_t, \
                                        time_key='time', keys_std=["cwp(gm-2)", 'toa_down_flux_all_lw(Wm-2)', 'toa_down_flux_clear_lw(Wm-2)', \
       'toa_down_flux_all_sw(Wm-2)', 'toa_up_flux_all_lw(Wm-2)', \
       'toa_up_flux_clear_lw(Wm-2)', 'toa_up_flux_all_sw(Wm-2)', \
       'toa_down_flux_clear_sw(Wm-2)', 'toa_up_flux_clear_sw(Wm-2)', \
       'toa_direct_flux_all_sw(Wm-2)', 'toa_direct_flux_clear_sw(Wm-2)', \
       'sfc_down_flux_all_lw(Wm-2)', 'sfc_down_flux_clear_lw(Wm-2)', \
       'sfc_down_flux_all_sw(Wm-2)', 'sfc_up_flux_all_lw(Wm-2)', \
       'sfc_up_flux_clear_lw(Wm-2)', 'sfc_up_flux_all_sw(Wm-2)', \
       'sfc_down_flux_clear_sw(Wm-2)', 'sfc_up_flux_clear_sw(Wm-2)', \
       'sfc_direct_flux_all_sw(Wm-2)', 'sfc_direct_flux_clear_sw(Wm-2)', 'wpi(1)', 'rl(um)', 'ri(um)', \
       'sfc_heating_rate_all_lw(K/day)', 'sfc_heating_rate_all_sw(K/day)', 'sfc_heating_rate_clear_lw(K/day)' ,'sfc_heating_rate_clear_sw(K/day)', \
       'sfc_heating_rate(K/day)', 'sfc_heating_rate_lw(K/day)', 'sfc_heating_rate_sw(K/day)', \
       'sfc_cre(Wm-2)', 'sfc_cre_lw(Wm-2)', 'sfc_cre_sw(Wm-2)'])
    if filter_invalid:
        results.to_csv("{}_AVG_{}_FILTER.csv".format(fname_out, int(delta_t)), index=False)
    else:
        results.to_csv("{}_AVG_{}.csv".format(fname_out, int(delta_t)))
        
if __name__ == '__main__':
    os.chdir("/home/phi.richter")
    #path_csv = "/home/phi.richter"
    #path_av = "/home/phi.richter"
    delta_t = 60
    files = ["RRTMG_Cloudnet_half_level.csv"]
    for file_ in [0]:#files:
        average("RRTMG_half_level_aot_0_05_albedo_clear_sky/RRTMG_results.csv", \
                delta_t, \
                "RRTMG_half_level_aot_0_05_albedo_clear_sky/RRTMG_results", \
                filter_invalid=True)
        average("RRTMG_half_level_aot_0_05_albedo_clear_sky/RRTMG_results.csv", \
                delta_t, \
                "RRTMG_half_level_aot_0_05_albedo_clear_sky/RRTMG_results", \
                filter_invalid=False)        
