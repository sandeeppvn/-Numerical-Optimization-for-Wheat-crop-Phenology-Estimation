'''
This script is to generate rasters of calculated APTT values.
Input: config.ini containing necessary paths and values
Output: APTT calculated Raster .tif files.

Details:
The raster file is a stack of scalar APTT values of dimension : Number of days x Number of Pixels in an image
Two raster files are calculated, one for vegitative index and one for representative index.
'''

import os
import numpy as np
from tqdm import tqdm
import rasterio
from Phenology_Utils import initialize_rasters

# Configparser
import configparser
config = configparser.ConfigParser()
config.read('config.ini')

# Initialize the path of Daymet data
daymet_path = config['PATHS']['daymet_path']
output_path = config['PATHS']['output_path']

# Initialize the Vegetation phenology parameters for APTT
Tmin_Veg, Topt_Veg, Tmax_Veg, P_critical_Veg, P_optimum_Veg = float(config['VEG']['Tmin']), float(config['VEG']['Topt']), float(config['VEG']['Tmax']), float(config['VEG']['P_critical']), float(config['VEG']['P_optimum'])

# Initialize the Reproductive phenology parameters for APTT
Tmin_Rep, Topt_Rep, Tmax_Rep, P_critical_Rep, P_optimum_Rep = float(config['REP']['Tmin']), float(config['REP']['Topt']), float(config['REP']['Tmax']), float(config['REP']['P_critical']), float(config['REP']['P_optimum'])

# Alpha values for APTT
alpha_Veg = np.log(2)/np.log((Tmax_Veg-Tmin_Veg)/(Topt_Veg-Tmin_Veg))
alpha_Rep = np.log(2)/np.log((Tmax_Rep-Tmin_Rep)/(Topt_Rep-Tmin_Rep))

m = 3
alpha_photo_Veg = np.log(2)/np.log((P_critical_Veg-P_optimum_Veg)/m + 1)
alpha_photo_Rep = np.log(2)/np.log((P_critical_Rep-P_optimum_Rep)/m + 1)

# Obtain the year of interest
harvest_year = int(config['INFO']['year'])

for year in tqdm(range(harvest_year-1,harvest_year)):

    tmin_path = os.path.join(daymet_path, 'daymet_data_tmin', 'tmin_'+str(year)+'.tif')
    tmax_path = os.path.join(daymet_path, 'daymet_data_tmax', 'tmax_'+str(year)+'.tif')
    dayl_path = os.path.join(daymet_path, 'daymet_data_dayl', 'daylength_'+str(year)+'.tif')
    
    outfile_Veg = os.path.join(output_path, 'APTT_winterwheat_vegetative_phase_'+str(year)+'.tif')
    outfile_Rep = os.path.join(output_path, 'APTT_winterwheat_reproductive_phase_'+str(year)+'.tif')

    tmin, tmax, t, dayl, tot_rows, tot_cols = initialize_rasters(tmin_path, tmax_path, dayl_path)

    # Initialize the APTT arrays
    ft1=np.copy(t)
    ft1[t>=Tmax_Veg]=0
    ft1[t<=Tmin_Veg]=0        
    ft1[(t<Tmax_Veg) & (t>Tmin_Veg)]=((2 * (t-Tmin_Veg)**alpha_Veg * (Topt_Veg-Tmin_Veg)**alpha_Veg - (t-Tmin_Veg)**(2*alpha_Veg))/((Topt_Veg-Tmin_Veg)**(2*alpha_Veg)))[(t<Tmax_Veg) & (t>Tmin_Veg)]
    ft1[np.where(ft1<0)]=0 

    ft2=np.copy(t)
    ft2[t>=Tmax_Rep]=0
    ft2[t<=Tmin_Rep]=0        
    ft2[(t<Tmax_Rep) & (t>Tmin_Rep)]=((2 * (t-Tmin_Rep)**alpha_Rep * (Topt_Rep-Tmin_Rep)**alpha_Rep - (t-Tmin_Rep)**(2*alpha_Rep))/((Topt_Rep-Tmin_Rep)**(2*alpha_Rep)))[(t<Tmax_Rep) & (t>Tmin_Rep)]
    ft2[np.where(ft2<0)]=0 

    c=dayl
    fp1=np.copy(c)
    fp1[c>=P_critical_Veg]=0
    fp1[c<=P_optimum_Veg]=1        
    fp1[(c<P_critical_Veg) & (c>P_optimum_Veg)]=((((c-P_optimum_Veg)/3 + 1)*((P_critical_Veg-c)/(P_critical_Veg-P_optimum_Veg))**((P_critical_Veg-P_optimum_Veg)/3))**alpha_photo_Veg)[(c<P_critical_Veg) & (c>P_optimum_Veg)]
    fp1[np.where(fp1<0)]=0    

    fp2=np.copy(c)
    fp2[c>=P_critical_Rep]=0
    fp2[c<=P_optimum_Rep]=1      
    fp2[(c<P_critical_Rep) & (c>P_optimum_Rep)]=((((c-P_optimum_Rep)/3 + 1)*((P_critical_Rep-c)/(P_critical_Rep-P_optimum_Rep))**((P_critical_Rep-P_optimum_Rep)/3))**alpha_photo_Rep)[(c<P_critical_Rep) & (c>P_optimum_Rep)]
    fp2[np.where(fp2<0)]=0  

    fpt1=ft1*fp1
    fpt2=ft2*fp2

    src=rasterio.open(tmin_path)
    profile=src.profile
    profile.update(count=fpt1.shape[0])

    with rasterio.open(outfile_Veg, 'w', **profile) as dst:
        for i in range(fpt1.shape[0]):
            dst.write(fpt1[i,:,:].astype(rasterio.float32), i+1)
    
    with rasterio.open(outfile_Rep, 'w', **profile) as dst:
        for i in range(fpt2.shape[0]):
            dst.write(fpt2[i,:,:].astype(rasterio.float32), i+1)






