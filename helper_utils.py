import folium
import pandas as pd
import os
import pickle
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
from folium.plugins import HeatMap
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb

def save_to_parquet(data, output_file_path):
    '''
    Save the data to a parquet file
    data: pd.DataFrame
        Data to save
    output_file_path: str
        Path to save the data
    '''
    data.to_parquet(output_file_path, engine='pyarrow', compression='snappy')

def load_raw_data(parquet_file_path):
    '''
    Load the raw data file
    parquet_file_path: str
        Path to the raw data file
    
    Returns
    -------
    raw_data: pd.DataFrame
        Raw data
    '''
    return pd.read_parquet(parquet_file_path)

def load_port_info():
    '''
    Load the port information file

    Returns
    -------
    port_info: list
        List of dictionaries containing port information
    '''
    port_info = pickle.load(open(os.path.join('data','extra_data','global_port_info.pkl'),'rb'))
    return port_info

def load_port_polygons():
    '''
    Load the port polygons file

    Returns
    -------
    port_gdf: gpd.GeoDataFrame
        GeoDataFrame containing port polygons
    '''
    port_gdf = pickle.load(open(os.path.join('data','extra_data','port_gdf.pkl'),'rb'))
    return port_gdf


def plot_one_trajectory(group,save_path=None):
    """
    Plot one trajectory onto a folium map
    
    group: pd.DataFrame
        Trajectory data
    save_path: str
        Path to save the plot if None the plot will not be saved
    """
    # Create a map centered on the mean latitude and longitude
    folium_map = folium.Map(location=[group['latitude'].mean(), group['longitude'].mean()], zoom_start=5)
    
    # Add a marker for each row in the filtered data
    for index, row in group.iterrows():
        # Create a marker with a popup label showing the MMSI
        folium.CircleMarker([row['latitude'], row['longitude']],
                            color='red',
                            radius=3,
                            ).add_to(folium_map)
    
    # Save the map to an HTML file
    if save_path is not None:
        folium_map.save(save_path)
    
    return folium_map


# -------------------------------------------------------------------------------------------
# ------------------------------------  Extra    --------------------------------------------
# -------------------------------------------------------------------------------------------

def analyze_trajectories_in_region(data, 
                                   ports_df, 
                                   port_name=None, 
                                   center_coords=None, 
                                   offset=1.0,
                                   start_time=None, 
                                   end_time=None
                                   ):
    """
    Analyze vessel trajectories based on a port name or geographic coordinates.
    
    Parameters:
        data (pd.DataFrame): Trajectory data with columns ['latitude', 'longitude', 'anonymized_mmsi'].
        ports_df (pd.DataFrame): Port data with columns ['port_name', 'polygon'].
        port_name (str): Name of the port to analyze. If None, use center_coords.
        center_coords (tuple): (latitude, longitude) for a custom location. Used if port_name is None.
        offset (float): Bounding box offset in degrees (default is 1 degree).
    
    Returns:
        dict: Contains vessel trajectory counts and Folium map.
    """
    # Determine bounding box
    if port_name:
        port_row = ports_df[ports_df["port_name"] == port_name]
        if port_row.empty:
            raise ValueError(f"Port '{port_name}' not found in ports DataFrame.")
        port_polygon = port_row["polygon"].iloc[0]
        center_coords = port_polygon.centroid.coords[0]
    
    if not center_coords:
        raise ValueError("Either 'port_name' or 'center_coords' must be provided.")
    
    lon, lat = center_coords
    bounding_box = {
        "min_lat": lat - offset,
        "max_lat": lat + offset,
        "min_lon": lon - offset,
        "max_lon": lon + offset,
    }
    
    # Filter trajectories in the bounding box
    filtered_data = data[
        (data["latitude"] >= bounding_box["min_lat"]) &
        (data["latitude"] <= bounding_box["max_lat"]) &
        (data["longitude"] >= bounding_box["min_lon"]) &
        (data["longitude"] <= bounding_box["max_lon"])
    ]
    
    if start_time is not None and end_time is not None:
        filtered_data = filtered_data[
            (filtered_data['time_of_position'] >= pd.to_datetime(start_time)) &
            (filtered_data['time_of_position'] <= pd.to_datetime(end_time))
        ]
    
    if filtered_data.empty:
        return {},  None

    # Identify distinct trajectories
    trajectories = filtered_data.groupby("anonymized_mmsi")[["latitude", "longitude"]].apply(
        lambda df: list(df.itertuples(index=False, name=None))
    )
    distinct_trajectories = {vessel: set(traj) for vessel, traj in trajectories.items()}
    trajectory_counts = {vessel: len(traj) for vessel, traj in distinct_trajectories.items()}

    # Create Folium map
    folium_map = folium.Map(location=[lat, lon], zoom_start=4)
    
    # Highlight bounding box
    bounding_box_polygon = [
        (bounding_box["min_lat"], bounding_box["min_lon"]),
        (bounding_box["min_lat"], bounding_box["max_lon"]),
        (bounding_box["max_lat"], bounding_box["max_lon"]),
        (bounding_box["max_lat"], bounding_box["min_lon"]),
        (bounding_box["min_lat"], bounding_box["min_lon"]),
    ]
    folium.Polygon(
        locations=bounding_box_polygon,
        color="red",
        fill=True,
        fill_opacity=0.3
    ).add_to(folium_map)
    
    # Plot trajectories
    for vessel, points in trajectories.items():
        folium.PolyLine(points, color="blue", weight=2.5, opacity=0.8).add_to(folium_map)
    
    return trajectory_counts,  folium_map

