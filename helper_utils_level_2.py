import folium
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Identify waiting periods
def detect_waiting(group, min_wait_time, max_speed):
    group['waiting_near_port'] = False
    near_port_periods = group[group['near_port']]
    if not near_port_periods.empty:
        group['waiting_near_port'] = (
            (group['near_port']) &
            (group['speed'] < max_speed) &  # Vessel stationary
            (group['time_of_position'].diff() >= min_wait_time)
        )
    return group.reset_index(drop=True)
    
    
def detect_waiting_near_ports(port_gdf: gpd.GeoDataFrame, 
                              anonymized_mmsi: str, 
                              buffer_distance: float,
                              min_wait_time: float, 
                              max_speed: float,
                              map_save_directory: str = None, 
                              data: pd.DataFrame= None):
    """
    Detect vessels waiting near ports (within a buffer zone) based on stationary behavior.
    Args:
        data: AIS data as a pandas DataFrame, including `latitude`, `longitude`, `speed`, and `time_of_position`.
        port_gdf: GeoDataFrame of ports with polygons representing port areas.
        min_wait_time: Minimum duration to classify as waiting.
        buffer_distance: Distance in meters for creating a buffer around port polygons.
    Returns:
        DataFrame with detected waiting events and a `waiting_near_port` column.
    """
    if data is not None:
        # Filter data for the selected vessel
        data = data[data["anonymized_mmsi"] == anonymized_mmsi]
    
    # Create a buffer zone around port polygons
    port_buffer = port_gdf.copy()
    port_buffer['polygon'] = port_gdf.geometry.buffer(buffer_distance)

    # Convert AIS data to GeoDataFrame
    ais_gdf = gpd.GeoDataFrame(
        data,
        geometry=[Point(xy) for xy in zip(data['longitude'], data['latitude'])],
        crs=port_gdf.crs
    )
    
    # Spatial join to check if vessel locations fall within port buffer zones
    ais_gdf['near_port'] = ais_gdf.geometry.apply(lambda geom: any(port_buffer.contains(geom)))

    # Sort by vessel and time
    ais_gdf.sort_values(by=['anonymized_mmsi', 'time_of_position'], inplace=True)

    ais_gdf = ais_gdf.groupby('anonymized_mmsi').apply(detect_waiting, min_wait_time= min_wait_time, max_speed= max_speed)
    ais_gdf.reset_index(drop=True, inplace=True)

    # Extract waiting events
    waiting_events = ais_gdf[ais_gdf['waiting_near_port']].copy()
    waiting_trajectory = ais_gdf[ais_gdf['near_port']].copy()
    waiting_events['duration'] = waiting_events['time_of_position'].diff()
    
    # Check if there are any waiting events
    if waiting_events.empty:
        print("No vessels detected as waiting near ports.")
        return waiting_events, None
    
    # Create a folium map centered around the first detected vessel's location
    map_center = [waiting_events['latitude'].iloc[0], waiting_events['longitude'].iloc[0]]
    m = folium.Map(location=map_center, zoom_start=2)

    # Add markers for each detected vessel
    for _, row in waiting_events.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Vessel: {row['anonymized_mmsi']}<br>Duration: {row['duration']}<br>Speed: {row['speed']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
        
    
    # Add a marker for each row in the filtered data
    for index, row in waiting_trajectory.iterrows():
        # Create a marker with a popup label showing the MMSI
        folium.CircleMarker([row['latitude'], row['longitude']],
                            color='red',
                            radius=3,
                            ).add_to(m)
        
    # Add the buffer zones to the map as polygons
    for _, buffer_row in port_buffer.iterrows():
        # Extract the buffered polygon geometry
        if buffer_row['polygon'].geom_type == 'Polygon':
            coordinates = [[point[1], point[0]] for point in buffer_row['polygon'].exterior.coords]
            folium.Polygon(
                locations=coordinates,
                color='green',
                fill=True,
                fill_opacity=0.2,
                popup="Port Detection Area"
            ).add_to(m)
        elif buffer_row['polygon'].geom_type == 'MultiPolygon':
            for poly in buffer_row['polygon']:
                coordinates = [[point[1], point[0]] for point in poly.exterior.coords]
                folium.Polygon(
                    locations=coordinates,
                    color='green',
                    fill=True,
                    fill_opacity=0.2,
                    popup="Port Detection Area"
                ).add_to(m)

    if map_save_directory != None:
        # Save the map to an HTML file
        m.save(f"detected_vessels_{anonymized_mmsi}.html")
    
    return waiting_events, m


def detect_visits_near_port(group):
    group['visit_id'] = (group['near_port'] & ~group['near_port'].fillna(False)).cumsum()
    return group.reset_index(drop=True)


def detect_unserved_movement_and_time_based(group, min_stay_time, max_speed_threshold, min_speed_change):
    group['unserved_visit_near_port'] = False
    group['entry'] = group['near_port'] & ~group['near_port'].shift(1).fillna(False)
    group['exit'] = ~group['near_port'] & group['near_port'].shift(1).fillna(False)

    # Identify visits based on entry and exit
    visits = group[group['entry'] | group['exit']].copy()
    visits['duration'] = visits['time_of_position'].diff().where(visits['exit'])
    visits['max_speed'] = visits['speed'].rolling(2).max()
    visits['speed_change'] = visits['speed'].diff().abs()  # Calculate speed change

    for i, visit in visits.iterrows():
        
        # Unserved if time is less than min_stay_time or movement indicates no stopping
        if visit['entry'] and (
            visit['duration'] < min_stay_time or  # Stayed too briefly
            visit['max_speed'] > max_speed_threshold or  # Exceeded allowed max speed
            visit['speed_change'] > min_speed_change  # Movement change too abrupt
        ):
            group.loc[i, 'unserved_visit_near_port'] = True

    return group.reset_index(drop=True)


def detect_unserved_visits_near_ports(  port_gdf: gpd.GeoDataFrame, 
                                        min_stay_time: pd.Timedelta, 
                                        buffer_distance: float,
                                        max_speed_threshold: float,
                                        min_speed_change: float,
                                        anonymized_mmsi: str,
                                        map_save_directory: str = None,                                         
                                        data: pd.DataFrame= None):
    """
    Detect unserved visits near ports (within buffer zones).
    Args:
        data: AIS data as a pandas DataFrame, including `latitude`, `longitude`, and `time_of_position`.
        port_gdf: GeoDataFrame of ports with polygons representing port areas.
        min_stay_time: Minimum duration to classify a visit as served.
        buffer_distance: Distance in meters for creating a buffer around port polygons.
    Returns:
        DataFrame with detected unserved visits and a `unserved_visit_near_port` column.
    """
    
    if data is not None:
        # Filter data for the selected vessel
        data = data[data["anonymized_mmsi"] == anonymized_mmsi]
        
    # Create a buffer zone around port polygons
    port_buffer = port_gdf.copy()
    port_buffer['polygon'] = port_gdf.geometry.buffer(buffer_distance)

    # Convert AIS data to GeoDataFrame
    ais_gdf = gpd.GeoDataFrame(
        data,
        geometry=[Point(xy) for xy in zip(data['longitude'], data['latitude'])],
        crs=port_gdf.crs
    )

    # Spatial join to check if vessel coordinates fall within port buffer zones
    ais_gdf['near_port'] = ais_gdf.geometry.apply(lambda geom: any(port_buffer.contains(geom)))

    # Group data by vessel and detect entry points (distinct visits)
    ais_gdf = ais_gdf.groupby('anonymized_mmsi').apply(detect_visits_near_port)

    # Reset index after grouping to avoid index and column conflicts
    ais_gdf.reset_index(drop=True, inplace=True)
    
    ais_gdf = ais_gdf.groupby(['anonymized_mmsi',"visit_id"]).apply(detect_unserved_movement_and_time_based, 
                                                       min_stay_time= min_stay_time, 
                                                       max_speed_threshold=max_speed_threshold,
                                                       min_speed_change=min_speed_change)
    ais_gdf.reset_index(drop=True, inplace=True)

    # Extract unserved visit events
    unserved_visits = ais_gdf[ais_gdf['unserved_visit_near_port']].copy()

    # Check if there are any waiting events
    if unserved_visits.empty:
        print("No vessels detected as unserved.")
        return unserved_visits, None
    
    # Create a folium map centered around the first detected vessel's location
    map_center = [unserved_visits['latitude'].iloc[0], unserved_visits['longitude'].iloc[0]]
    m = folium.Map(location=map_center, zoom_start=2)
    
    # Add markers for each detected vessel
    for _, row in unserved_visits.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Vessel: {row['anonymized_mmsi']}<br>Speed: {row['speed']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
        
    
    # Add a marker for each row in the filtered data
    for index, row in unserved_visits.iterrows():
        # Create a marker with a popup label showing the MMSI
        folium.CircleMarker([row['latitude'], row['longitude']],
                            color='red',
                            radius=3,
                            ).add_to(m)
        
    # Add the buffer zones to the map as polygons
    for _, buffer_row in port_buffer.iterrows():
        # Extract the buffered polygon geometry
        if buffer_row['polygon'].geom_type == 'Polygon':
            coordinates = [[point[1], point[0]] for point in buffer_row['polygon'].exterior.coords]
            folium.Polygon(
                locations=coordinates,
                color='green',
                fill=True,
                fill_opacity=0.2,
                popup="Port Detection Area"
            ).add_to(m)
        elif buffer_row['polygon'].geom_type == 'MultiPolygon':
            for poly in buffer_row['polygon']:
                coordinates = [[point[1], point[0]] for point in poly.exterior.coords]
                folium.Polygon(
                    locations=coordinates,
                    color='green',
                    fill=True,
                    fill_opacity=0.2,
                    popup="Port Detection Area"
                ).add_to(m)

    if map_save_directory != None:
        # Save the map to an HTML file
        m.save(map_save_directory)
    
    return unserved_visits, m