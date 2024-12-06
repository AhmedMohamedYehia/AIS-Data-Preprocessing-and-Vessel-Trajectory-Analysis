import folium
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def detect_idle_states(data: pd.DataFrame, speed_threshold: float = 0.5) -> pd.DataFrame:
    """
    Detects idle states where vessel speed is below the threshold.
    Adds a column `is_idle` indicating the state.
    """
    print("Number of all instnces: " + str(data.shape[0]))
    
    data = data.assign(is_idle=lambda df: df['speed'] < speed_threshold)
    data = data[data["is_idle"]==True]
    
    print("Number of Idle instnces: " + str(data.shape[0]))
    
    return data


def convert_to_gpd(data: pd.DataFrame, port_gdf: gpd.GeoDataFrame)-> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        data,
        geometry=[Point(xy) for xy in zip(data['longitude'], data['latitude'])],
        crs=port_gdf.crs
    )
    
    
def is_in_port(port_gdf: gpd.GeoDataFrame, ais_gdf: gpd.GeoDataFrame)-> gpd.GeoDataFrame:
    return ais_gdf.assign(
        in_port=lambda df: df.geometry.apply(lambda geom: any(port_gdf.contains(geom))),
        port_name=lambda df: df.geometry.apply(
            lambda geom: next(
                (port_name for port_name in port_gdf.loc[port_gdf.contains(geom), 'port_name'].values),
                None
            )
        )
    )


def plot_vessel_trajectory_ports_map(data: pd.DataFrame,visited_port_gdf:gpd.GeoDataFrame, save_directory: str= None)-> None:

    # Create a Folium map
    map = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=5)
    
    # Add a marker for each row in the filtered data
    for index, row in data.iterrows():
        # Create a marker with a popup label showing the MMSI
        folium.CircleMarker([row['latitude'], row['longitude']],
                            color='red',
                            radius=1,
                            ).add_to(map)

    # Add polygons to the map
    for _, row in visited_port_gdf.iterrows():
        folium.GeoJson(row['polygon'], name=row['port_name'], popup=row['port_name']).add_to(map)  # Explicitly access the 'geometry' column

    if save_directory!= None:
        map.save(save_directory)
        print("Map saved at: "+ save_directory)
    
    return map


def detect_visits_in_port(group):
    group['visit_id'] = (group['in_port'] & ~group['in_port'].shift(1).fillna(False)).cumsum()
    return group.reset_index(drop=True)


def get_visits_summary(ais_gdf: gpd.GeoDataFrame):
    # Sort data by vessel and time_of_position
    ais_gdf = ais_gdf.sort_values(by=['anonymized_mmsi', 'time_of_position'])

    # Group data by vessel and detect entry points (distinct visits)
    ais_gdf = ais_gdf.groupby('anonymized_mmsi').apply(detect_visits_in_port)

    # Reset index after grouping to avoid index and column conflicts
    ais_gdf.reset_index(drop=True, inplace=True)

    # Summarize distinct visits per vessel with port names
    visit_summary = (
        ais_gdf[ais_gdf['in_port']]
        .groupby(['anonymized_mmsi', 'visit_id'])
        .agg({'port_name': 'first'})  # Extract the first (unique) port name per visit
        .reset_index()
        .groupby('anonymized_mmsi')
        .agg(
            distinct_visits=('visit_id', 'nunique'),
            port_names=('port_name', lambda x: list(set(x)))  # List of unique port names per vessel
        )
        .reset_index()
    )
    return ais_gdf, visit_summary


def visualize_vessel_trajectory_and_ports(port_gdf: gpd.GeoDataFrame, 
                                          anonymized_mmsi: str, 
                                          map_save_directory: str= None,
                                          data: pd.DataFrame= None):
    """
    Visualize the vessel trajectory along with port polygons using Matplotlib and Folium.
    """
    if data is not None:
        # Filter data for the selected vessel
        data = data[data["anonymized_mmsi"] == anonymized_mmsi]

    # Convert AIS data to GeoDataFrame
    ais_gdf = convert_to_gpd(data, port_gdf)

    # Annotate whether the vessel is in a port
    ais_gdf = is_in_port(port_gdf, ais_gdf)
    
    # Get visits summary
    ais_gdf_output, visit_summary = get_visits_summary(ais_gdf)

    # Extract the visited ports
    visited_ports = ais_gdf.loc[ais_gdf['in_port'], 'port_name'].dropna().unique()
    visited_port_gdf = port_gdf[port_gdf["port_name"].isin(visited_ports)]
    
    # save Folium map
    map = plot_vessel_trajectory_ports_map(data, visited_port_gdf, map_save_directory)
    
    return ais_gdf_output, visit_summary, map