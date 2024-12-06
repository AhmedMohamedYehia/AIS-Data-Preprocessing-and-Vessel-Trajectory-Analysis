import pickle
import os
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from helper_utils import load_raw_data,save_to_parquet
import warnings
warnings.filterwarnings("ignore")

def remove_NA(data):
    # Drop rows with missing values
    print('**************')
    print('*** STEP 1 ***')
    print('Dropping rows with missing values...')
    data = data.dropna()
    print('Rows after droping NA: ',data.shape[0])
    print('**************')
    print()
    return data


def remove_invalid_lat_and_lon(data):
    # Remove rows with invalid latitudes and longitudes
    print('**************')
    print('*** STEP 2 ***')
    print('Removing rows with invalid latitudes and longitudes...')
    data =  data[(data['latitude']>-90.) & (data['latitude']<90.)&(data['longitude']>-180.) & (data['longitude']<180.)]
    print('Rows after droping invalid rows with latitudes and longitudes: ',data.shape[0])
    print('**************')
    print()
    return data


def remove_invalid_speed(data):
    # Remove rows with invalid speed
    print('**************')
    print('*** STEP 3 ***')
    print('Removing rows with invalid speed...')
    data =  data[(data['speed']<=60.) & (data['speed']>0.)]    
    print('Rows after droping invalid speed: ',data.shape[0])
    print('**************')
    print()
    return data


def convert_to_datetime(data):
    print('**************')
    print('*** STEP 4 ***')
    print('Converting "time_of_position" column type to datetime...')
    data['time_of_position'] = pd.to_datetime(data['time_of_position'], errors='coerce')
    print('Data Type after converting: ',data['time_of_position'].dtype)
    print('**************')
    print()
    return data







if __name__ == '__main__':

    print('********************************')
    print('Processing raw data ----- TASK 1')
    print('********************************')
    print()
    print()
    print()
    
    # Raw data file path
    raw_data_file_path = 'data/raw_data/raw_data.parquet'
    
    # Output directory, file name, and save path
    output_dir = 'data/Preprocessed'
    output_file_name = 'preprocessed.parquet'
    save_path = os.path.join(output_dir, output_file_name)


    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print('**************')
    print('Loading raw data...')
    data = load_raw_data(raw_data_file_path)
    print('Number of rows',data.shape[0])
    print('**************')
    print()
    data = remove_NA(data)
    data = remove_invalid_lat_and_lon(data)
    data = remove_invalid_speed(data)
    

    print('**************')
    print(f'Saving cleaned data to parquet to {save_path}...')
    save_to_parquet(data,save_path)
    print('**************')