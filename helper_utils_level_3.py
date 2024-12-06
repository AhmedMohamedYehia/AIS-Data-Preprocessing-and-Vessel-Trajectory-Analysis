import folium
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
from folium.plugins import HeatMap
def is_within_buffer(point, center, radius):
    return point.distance(center) <= radius

#  temporal
def calculate_average_duration_and_map(data: pd.DataFrame, 
                                       lat: float, 
                                       lon: float, 
                                       buffer_radius: float, 
                                       anonymized_mmsis: list = None,
                                       map_save_directory: str = None):  
    
    if anonymized_mmsis is not None:
        # Filter data for the selected vessels
        data = data[data["anonymized_mmsi"].isin(anonymized_mmsis)]  
        
    # Create circular buffer geometry
    center = Point(lon, lat)  # Longitude first for Shapely
    circle = center.buffer(buffer_radius / 111.32)  # Approx: 1° = 111.32 km

    # Convert data to GeoDataFrame
    data['geometry'] = data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    ais_gdf = gpd.GeoDataFrame(data, geometry='geometry', crs="EPSG:4326")

    # Filter vessels passing through the circular area
    within_circle = ais_gdf[ais_gdf.geometry.within(circle)]
    within_circle['geometry'].apply(lambda geom: is_within_buffer(geom, center, buffer_radius / 111.32))


    # Sort data by vessel and time
    within_circle = within_circle.sort_values(['anonymized_mmsi', 'time_of_position'])

    # Define entry and exit points
    within_circle['entry'] = ~within_circle['geometry'].shift(1).within(circle) & within_circle['geometry'].within(circle)
    within_circle['exit'] = within_circle['geometry'].within(circle) & ~within_circle['geometry'].shift(-1).within(circle)
    
    # Calculate durations and filter valid entries
    entry_times = within_circle.loc[within_circle['entry'], 'time_of_position'].reset_index(drop=True)
    exit_times = within_circle.loc[within_circle['exit'], 'time_of_position'].reset_index(drop=True)

    if len(entry_times) > 0 and len(exit_times) > 0:
        durations = exit_times - entry_times
        valid_durations = durations[durations > pd.Timedelta(0)]
        avg_duration = valid_durations.mean()
    else:
        avg_duration = pd.Timedelta(0)

    # Plot trajectories on a Folium map
    map_obj = folium.Map(location=[lat, lon], zoom_start=3)
    for mmsi, group in within_circle.groupby('anonymized_mmsi'):
        points = group.sort_values('time_of_position')['geometry'].apply(lambda x: (x.y, x.x)).tolist()
        folium.PolyLine(points, color="blue", weight=2.5, opacity=0.7).add_to(map_obj)
    
    folium.Circle(location=[lat, lon], radius=buffer_radius * 1000, color='red', fill=True, fill_opacity=0.3).add_to(map_obj)

    if map_save_directory != None:
        # Save the map to an HTML file
        map_obj.save(map_save_directory)
    
    return avg_duration, map_obj

# port peak times

def plot_peak_times_hour(peak_times, port_name, chosen_vessels):
    plt.figure(figsize=(10, 6))
    plt.bar(peak_times['hour'], peak_times['entry_count'], color='skyblue')
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.title(f'Peak Times for Port: {port_name}, chosen vessels {chosen_vessels}')
    plt.xticks(range(24))  
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
def plot_peak_times_day_of_week(peak_times, port_name, chosen_vessels):
    plt.figure(figsize=(10, 6))
    plt.bar(peak_times['day_of_week'], peak_times['entry_count'], color='skyblue')
    
    plt.xlabel('Day of week')
    plt.ylabel('Count')
    plt.title(f'Peak Times for Port: {port_name}, chosen vessels {chosen_vessels}')
    plt.xticks(range(7))  
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    
def plot_peak_times_month(peak_times, port_name, chosen_vessels):
    plt.figure(figsize=(10, 6))
    plt.bar(peak_times['month'], peak_times['entry_count'], color='skyblue')
    
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.title(f'Peak Times for Port: {port_name}, chosen vessels {chosen_vessels}')
    plt.xticks(range(12))  # Make sure the x-axis shows all 24 hours
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def identify_peak_times_at_port(data:pd.DataFrame, 
                                port_gdf: gpd.GeoDataFrame, 
                                buffer_radius:float,
                                port_name: str = None, 
                                anonymized_mmsis: list = None):
    
    if anonymized_mmsis is not None:
        # Filter data for the selected vessels
        data = data[data["anonymized_mmsi"].isin(anonymized_mmsis)]  
        
    # Find the port geometry
    port_geom = False
    if port_name is not None:
        port_geom = port_gdf[port_gdf['port_name'] == port_name].geometry.values[0]
    else: 
        port_geom = port_gdf.geometry.values[0]
        
    buffer_geom = port_geom.buffer(buffer_radius / 111.32)  # Approx: 1° = 111.32 km

    # Convert AIS data to GeoDataFrame
    data['geometry'] = data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    ais_gdf = gpd.GeoDataFrame(data, geometry='geometry', crs="EPSG:4326")

    # Filter vessels entering the buffer area
    ais_gdf['near_port'] = ais_gdf.geometry.apply(lambda geom: buffer_geom.contains(geom))
    port_entries = ais_gdf[ais_gdf['near_port']]

    # Calculate entry events and group by hour
    port_entries['hour'] = port_entries['time_of_position'].dt.hour
    peak_times_hour = port_entries.groupby('hour').size().reset_index(name='entry_count')
    peak_times_hour = peak_times_hour.sort_values(by='entry_count', ascending=False).reset_index(drop=True)
    
    port_entries['day_of_week'] = port_entries['time_of_position'].dt.day_of_week
    peak_times_day_of_week = port_entries.groupby('day_of_week').size().reset_index(name='entry_count')
    peak_times_day_of_week = peak_times_day_of_week.sort_values(by='entry_count', ascending=False).reset_index(drop=True)
    
    port_entries['month'] = port_entries['time_of_position'].dt.month
    peak_times_month = port_entries.groupby('month').size().reset_index(name='entry_count')
    peak_times_month = peak_times_month.sort_values(by='entry_count', ascending=False).reset_index(drop=True)
    
    plot_peak_times_hour(peak_times_hour, port_name, anonymized_mmsis)
    plot_peak_times_day_of_week(peak_times_day_of_week, port_name, anonymized_mmsis)
    plot_peak_times_month(peak_times_month, port_name, anonymized_mmsis)
    
    return peak_times_hour, peak_times_month, peak_times_day_of_week


#  Spatial Metrics:

def analyze_vessel_activity(data, 
                            anonymized_mmsis: list = None,
                            grid_size=0.01, 
                            threshold=10,
                            map_save_directory: str = None):
    """
    Analyzes vessel activity, calculates frequent areas, and identifies hotspots (e.g., anchorage zones).
    The analysis includes clustering, density calculation, and visualization of hotspots.
    
    Parameters:
    - data: DataFrame with latitude, longitude, and timestamps for vessel positions.
    - center_point: Tuple of (lat, long) for the center of interest (port or anchorage).
    - buffer_distance: Radius of interest around the center in meters.
    - grid_size: Size of the grid cells for density calculation.
    - threshold: Density threshold for identifying hotspots.
    - eps: DBSCAN parameter for clustering (distance threshold).
    - min_samples: DBSCAN parameter for clustering (minimum samples per cluster).
    
    Returns:
    - None. The function performs analysis and visualizes hotspots.
    """
    
    if anonymized_mmsis is not None:
        # Filter data for the selected vessels
        data = data[data["anonymized_mmsi"].isin(anonymized_mmsis)]  
        
    # Convert lat, long to geospatial points
    data['geometry'] = data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(data, geometry='geometry')
    
    # Create a grid
    x_min, y_min, x_max, y_max = gdf.total_bounds
    x_bins = np.arange(x_min, x_max, grid_size)
    y_bins = np.arange(y_min, y_max, grid_size)
    grid_counts = np.zeros((len(x_bins), len(y_bins)))
    
    # Calculate the density of vessels in each grid cell
    for _, row in gdf.iterrows():
        try:
            x_idx = np.clip(np.digitize(row.geometry.x, x_bins) - 1, 0, len(x_bins) - 1)
            y_idx = np.clip(np.digitize(row.geometry.y, y_bins) - 1, 0, len(y_bins) - 1)
            grid_counts[x_idx, y_idx] += 1
        except IndexError:
            continue  # Skip points outside the grid bounds

    # Identify hotspots (density > threshold)
    hotspot_coords = np.where(grid_counts > threshold)
    hotspot_points = [
        (
            y_bins[y_idx] + grid_size / 2,  # Center of the cell in y (latitude)
            x_bins[x_idx] + grid_size / 2  # Center of the cell in x (longitude)
        )
        for x_idx, y_idx in zip(hotspot_coords[0], hotspot_coords[1])
    ]
    
    # Create a Folium map
    m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=5)
    
    # Add trajectory points to map
    # for _, row in gdf.iterrows():
    #     folium.CircleMarker(
    #         location=(row.geometry.y, row.geometry.x),
    #         radius=2,
    #         color="green",
    #         fill=True,
    #         fill_opacity=0.6,
    #     ).add_to(m)
    
    # Add hotspots as heatmap
    HeatMap(
        hotspot_points,
        radius=20,
        blur=10,
        max_zoom=15
    ).add_to(m)
    
    if map_save_directory != None:
        # Save the map to an HTML file
        m.save(map_save_directory)
    
    return m
