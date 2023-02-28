import imp
from unittest import result
import rasterio
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


def initialize_rasters(tmin_path, tmax_path, dayl_path):
    
    raster1=rasterio.open(tmin_path)
    tmin=raster1.read()
    tmin=np.delete(tmin, list(range(181,365)), 0)

    raster2=rasterio.open(tmax_path)
    tmax=raster2.read()
    tmax=np.delete(tmax, list(range(181,365)), 0)

    t = (0.5*np.add(tmin,tmax))

    raster3=rasterio.open(dayl_path)
    dayl=raster3.read()/3600.0

    return (tmin, tmax, t, dayl, raster1.height, raster1.width)


def smooth_ndvi(year,path):
    state_avg_ndvi = pd.read_csv(os.path.join(path, 'state_average_wheat_NDVI_'+str(year-1)+'_'+str(year)+'.csv'), usecols=['year', 'doy', 'NDVI'])

    # Filter out data before 249th day of last year and after 249th day of year
    df = state_avg_ndvi[
        ((state_avg_ndvi.year == year-1) & (state_avg_ndvi.doy >= 249)) | 
        ((state_avg_ndvi.year == year) & (state_avg_ndvi.doy <= 249))
    ]
    
    ndvi = np.array(df['NDVI']).astype(float)
    doys = np.array(df['doy']+365*(df['year']-df.iloc[0]['year'])).astype(float)

    # The overhead is doys[0], hence we subtact the overhead from doys
    doys = doys - doys[0]

    fitted_ndvi, xsol, doys_cont = get_fitted_ndvi(ndvi, doys)

    dividing_index = get_dividing_index(fitted_ndvi)

    # Plotting
    plt.plot(doys, ndvi)
    plt.plot(doys, fitted_ndvi, 'r')
    plt.plot(doys[dividing_index], ndvi[dividing_index],'--rx')
    plt.show()

    return (ndvi, doys, fitted_ndvi, dividing_index)




def get_dividing_index(fitted_ndvi):
    '''
    This function returns the index of the day the second peak starts.
    This can be estimated as a function of troughs and crests.
    '''
    troughs, crests = get_troughs_and_crests(fitted_ndvi)

    # Figure out why
    if troughs[0] < crests[0] and len(troughs) > len(crests):
        troughs = troughs[1:]

    if len(troughs)==2:
        dividing_index = int(((2*troughs[0])+troughs[1])/3)
    elif len(crests)==2:
        dividing_index = int((crests[0]+crests[1])/2)
        
    # The dividing index is the index of the first trough after the first crest
    return dividing_index

from scipy import signal
def get_troughs_and_crests(fitted_ndvi):
    '''
    This function returns the troughs and crests of the fitted double logistic function.
    '''
    diffn = np.diff(fitted_ndvi)
    crests, crest_values = signal.find_peaks(diffn, height=0)
    troughs, trough_values = signal.find_peaks(-diffn, height=0)    



    return troughs, crests

from scipy.optimize import least_squares, curve_fit
def get_fitted_ndvi(ndvi,doys):
    '''
    This function uses the ndvis and doys to perform least square optimization and return a fitted double logistic function.
    Since the data contains two peaks, we create two functions and add them to get the overall double logistic function.
    ''' 

    # Initial parameters and bounds obtained through intuition
    init_params_peak1 = [0.8, -40.0, 0.6, -50.0, 2000, 4000, 4000]
    init_params_peak2 = [0.8, -140.0, 0.6, -150.0, 4000, 4000]

    # Bounds for the parameters
    lower_bounds_peak1 = [0.0, -100.0, 0.0, -100.0, 0.0, 0.0, 0.0]
    upper_bounds_peak1 = [1.5, -8.0, 0.9, 0.0, 3000, 5000, 5000]

    lower_bounds_peak2 = [0.05, -290.0, 0.1, -400.0, 2500, 2500]
    upper_bounds_peak2 = [1.5, -80.0, 0.9, -130.0, 7000, 7000]

    result = least_squares(
        loss_function, 
        init_params_peak1 + init_params_peak2, 
        bounds=(
            lower_bounds_peak1 + lower_bounds_peak2, 
            upper_bounds_peak1 + upper_bounds_peak2
        ), 
        max_nfev=100000,
        args=(doys, ndvi)
    )

    xsol = result.x

    # doys_cont is the continuous set of days spanning min to max doys
    doys_cont = np.arange(doys[0], doys[-1], 1)

    # Fit the double logistic function to the continuous doys with the optimized parameters
    fitted_ndvi = xsol[4] + (xsol[5]-xsol[4]) * (expit(xsol[0] * (doys_cont - xsol[1])) - expit(xsol[2] * (doys_cont - xsol[3])))

    return fitted_ndvi, xsol, doys_cont


def loss_function(params, doys, ndvi):
    '''
    This function returns the loss value for the double logistic function fitting between the ndvi and modeled ndvi from the doys and parameters.
    '''
    output = []
    losses = ndvi - two_peak_double_logistic(doys, params)
    for loss in losses:
        output.append(loss)
    return np.array(output).squeeze()

from scipy.special import expit
def two_peak_double_logistic(doys, params):
    '''
    This function returns the double logistic function for the given doys and parameters.
    '''
    t = np.array(doys)

    result_peak1 =  params[5]  * expit(params[0] * t + params[1]) - params[6]  * expit(params[2] * t + params[3] ) + params[4]
    result_peak2 = params[11]  * expit(params[7] * t + params[8]) - params[12] * expit(params[9] * t + params[10])
    return result_peak1 + result_peak2