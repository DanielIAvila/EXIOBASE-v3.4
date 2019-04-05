# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 16:28:17 2018

@author: DanielAvila, based on Arjan de Koning's code.

README:
1. To run this code, you should have the script in a folder. Within this folder create two new folders > raw & clean.
2. In the "raw" folder, you should unzip EXIOBASE v3.4 years (folder per year), which you can download from https://exiobase.eu/.
3. Run the code.

This code process Exiobase v3.4 (industry by industry/sector by sector) as time series (1995 - 2011). 
It saves, if specified, the files in a pickle format.
You will get then as result:
    direct_satelliteYEAR.pkl
    iotYEAR.pkl
    meYEAR.pkl
    mmYEAR.pkl
    mmb_finalYEAR.pkl
    mrYEAR.pkl
    mwYEAR.pkl
    myYEAR.pkl
    satelliteYEAR.pkl
    wYEAR.pkl
    x_outSYEAR.pkl
    YYEAR.pkl
"""

import os
import os.path
import csv
import pickle
import numpy as np
import pandas as pd
import gc

os.getcwd()

years = [1995, 2012]
raw_data_dir = os.path.join('..', 'data', 'raw')
clean_data_dir = os.path.join('..', 'data', 'clean')

variables = ['x_outS', 'mm'] # These variables were specifically named for the Circularity Gap Report for Circular Economy
                             # the second element in the list can be any of the available extensions in Exiobase, mm = materials

ind_cnt = 163 # Here you can change industry by industry, or sector by sector.
cntr_cnt = 49    
    
def read_file(filename, separated, header, index_col):
    with open(filename) as f:
        reader = pd.read_csv(f, sep=separated, header=header, index_col=index_col, encoding='utf-8')
        reader.astype(float)
    return reader

def read_pickle(directory, filename):
    """ """
    reader = pd.read_pickle(directory + '/' + str(filename) + '.pkl')
    return reader

def save_pickle(directory, filename, year, data):
    """  """
    data.to_pickle(directory + '/' + filename + str(year) + '.pkl')
  
def process_exiobase(raw_data_dir, clean_data_dir, year, condition=True):
    """  """
    
    iot_filename = 'A.txt'
    final_demand_filename = 'Y.txt'
    satellite_filename = 'F.txt'
    direct_satellite_filename = 'F_hh.txt'
    
    ind_cnt = 163 # Here you can change industry by industry, or sector by sector.
    cntr_cnt = 49
    
    factor_inputs_index = 23
    emissions_index = 446
    resources_index = 466
    energy_index = 470
    materials_index = 910
    water_index = 1104
    
    # create canonical filename
    full_raw_io_data_dir = os.path.join(raw_data_dir, 'IOT_' + str(year) + '_ixi')
    full_raw_ext_data_dir = os.path.join(full_raw_io_data_dir, 'satellite')
    full_iot_fn = os.path.join(full_raw_io_data_dir, iot_filename)
    full_final_demand_fn = os.path.join(full_raw_io_data_dir, final_demand_filename)
    full_satellite_fn = os.path.join(full_raw_ext_data_dir, satellite_filename)
    full_direct_satellite_fn = os.path.join(full_raw_ext_data_dir, direct_satellite_filename)
    
    # read files
    iot = read_file(full_iot_fn, '\t', [0,1], [0,1]) # A in Exiobase v3.4
    final_demands = read_file(full_final_demand_fn, '\t', [0,1], [0,1])
    satellite = read_file(full_satellite_fn, '\t', [0,1], [0])
    direct_satellite = read_file(full_direct_satellite_fn, '\t', [0,1], [0])
    
    # total output
    y_tot = final_demands.sum(1).rename('Ytot', inplace=True)  
    i = np.eye(ind_cnt * cntr_cnt)
    leontief = np.linalg.inv(i - iot)
    x = np.dot(leontief, y_tot) # array
    x_outS = pd.Series(x, index=iot.index.tolist()).rename('Xtot', inplace=True)
    x_out = pd.DataFrame(x)
    x_out['region'] = list(iot.index.get_level_values(0))
    x_out['sector'] = list(iot.index.get_level_values(1))
        
    # stressors
    m = satellite
    w = m.iloc[:factor_inputs_index, :] # value added 
    me = m.iloc[factor_inputs_index:emissions_index, :] # emissions
    mr = m.iloc[emissions_index:resources_index, :] # resources (land)
    my = m.iloc[resources_index:energy_index, :] # resources (energy)
    mm = m.iloc[energy_index:materials_index, :] # resources (materials)
    mw = m.iloc[materials_index:water_index, :] # resources (water)
 
    if condition == True:
        return iot, final_demands, x_out, x_outS, satellite, direct_satellite, w, me, mr, my, mm, mw
    if condition == False:
        save_pickle(clean_data_dir, 'iot', year, iot)
        save_pickle(clean_data_dir, 'Y', year, final_demands)
        save_pickle(clean_data_dir, 'x_outS', year, x_outS)
        save_pickle(clean_data_dir, 'satellite', year, satellite)
        save_pickle(clean_data_dir, 'direct_satellite', year, direct_satellite)
        save_pickle(clean_data_dir, 'w', year, w)
        save_pickle(clean_data_dir, 'me', year, me)
        save_pickle(clean_data_dir, 'mr', year, mr)
        save_pickle(clean_data_dir, 'my', year, my)
        save_pickle(clean_data_dir, 'mm', year, mm)
        save_pickle(clean_data_dir, 'mw', year, mw)

def resources_impact(data_output, data_resources, cntr_cnt, ind_cnt, clean_data_dir, year, condition=True):
    """ This function reads for a given year:
        - the total output (data_output)
        - the environmental extension (material_resources)
        - the country and sector/industry count (cntr_cnt & ind_cnt)
        It returns the:
        - the environmental coefficient (mmb)
        - the environmental requirement per industry/sector (mmb_final)"""
        
    inv_x = 1/data_output     # Inverse matrix to calculate A, A = technical coefficient matrix
    inv_x[inv_x == np.inf] =0 # It cleans all "inf" values and convert its to "0"
    inv_x = np.diag(inv_x)    # Diagonal of the previous matrix
    
    # Total (grouped) resources, based on intermediate resources (kilo tonnes)
    mmt = pd.DataFrame(data_resources.sum(0)).T            # Sums all rows and transpose it to a column
    mmi = data_resources.div(mmt.iloc[0], axis='columns')  # It divide each row by the total
    mmi.fillna(0, inplace=True)
    mmt = np.reshape(np.array(mmt), (cntr_cnt*ind_cnt, 1)) # It reshapes the array
    
    # Material coefficient (kt/M EUR), based on intermediate resources
    mmb= pd.DataFrame(np.transpose(np.dot(np.transpose(mmt), inv_x)), index=None, columns=['Value'])
    
    mmb_list = mmb['Value'].tolist()
    mmb_final = mmi.mul(mmb_list, axis=1)
    if condition == True:
        return mmb, mmb_final
    if condition == False:
        save_pickle(clean_data_dir, 'mmb', year, mmb)
        save_pickle(clean_data_dir, 'mmb_final', year, mmb_final)
        
def extension_impact(data_output, data_resources, cntr_cnt, ind_cnt, clean_data_dir, year, lab='', lab2='', 
                     condition=True):
    """ This function reads for a given year:
        - the total output (data_output)
        - the environmental extension (any)
        - the country and sector/industry count (cntr_cnt & ind_cnt)
        It returns the:
        - the environmental coefficient vector (mxb)
        - the environmental requirement per industry/sector matrix (mxb_final)"""
    
    inv_x = 1/data_output     # Inverse matrix to calculate A, A = technical coefficient matrix
    inv_x[inv_x == np.inf] =0 # It cleans all "inf" values and convert its to "0"
    inv_x = np.diag(inv_x)    # Diagonal of the previous matrix
    
    # Total (grouped) resources, based on intermediate resources (units)
    met = pd.DataFrame(data_resources.sum(0)).T           # Sums all rows and transpose it to a column
    mei = data_resources.div(met.iloc[0], axis='columns') # It divide each row by the total
    mei.fillna(0, inplace=True)
    met = np.reshape(np.array(met), (cntr_cnt*ind_cnt, 1)) # It reshapes the array
    
    # Extension coefficient vector (unit/M EUR), based on intermediate resources
    meb= pd.DataFrame(np.transpose(np.dot(np.transpose(met), inv_x)), index=None, columns=['Value'])
    
    meb_list = meb['Value'].tolist()
    meb_final = mei.mul(meb_list, axis=1) # Extension coefficient matrix (unit/ M EUR), based on intermediate resources
    if condition == True:
        return meb, meb_final
    if condition == False:
        save_pickle(clean_data_dir, lab, year, meb)
        save_pickle(clean_data_dir, lab2, year, meb_final)
        
def file_names(yearOne, yearTwo, variables):
    """ This function creates the filenames to use for the processed files (as output)."""
    
    filenames = []
    for i in np.arange(yearOne, yearTwo):
        file_x, file_mm = variables[0] + str(i) , variables[1] + str(i)
        filenames.append([file_x, file_mm]) 
        
    return filenames        

def time_Series(yearOne, yearTwo, raw_data_dir, clean_data_dir, filenames, cntr_cnt, ind_cnt):    
    """ This function process all EXIOBASE as time series (from any given year to any other given year).""" 
    for i, j in enumerate(np.arange(yearOne, yearTwo)):
        process_exiobase(raw_data_dir, clean_data_dir, j, condition=False)
        x_out = read_pickle(clean_data_dir,filenames[i][0])
        mm = read_pickle(clean_data_dir,filenames[i][1])
        resources_impact(x_out, mm, cntr_cnt, ind_cnt, clean_data_dir, j, condition=False)    

        
filenames = file_names(years[0], years[1], variables)
time_Series(1995, 2012, raw_data_dir, clean_data_dir, filenames, cntr_cnt, ind_cnt)
