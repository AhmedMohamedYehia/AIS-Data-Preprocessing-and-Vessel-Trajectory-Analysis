import folium
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb


def visualize_waiting_times(waiting_summary: pd.DataFrame, port_name:str = None, save_path: str = None):
    """
    Visualize the waiting times and idle periods for vessels.

    Args:
        waiting_summary (pd.DataFrame): Summary DataFrame with columns:
                                        - anonymized_mmsi
                                        - total_waiting_time
                                        - num_idle_periods
        save_path (str): Path to save the visualization. (Optional)

    Returns:
        None: Displays the chart and optionally saves it.
    """
    # Convert total waiting time to days for visualization
    waiting_summary['waiting_time_days'] = waiting_summary['total_waiting_time'].dt.total_seconds() / (24 * 3600)

    # Sort by waiting time for better visualization
    waiting_summary = waiting_summary.sort_values(by='waiting_time_days', ascending=False)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Bar chart for waiting time
    bar_width = 0.4
    vessels = waiting_summary['anonymized_mmsi']
    waiting_times = waiting_summary['waiting_time_days']
    idle_periods = waiting_summary['num_idle_periods']

    ax1.bar(vessels, waiting_times, color='blue', label='Waiting Time (days)', width=bar_width)
    ax1.set_ylabel('Waiting Time (days)', color='blue')
    ax1.set_xlabel('Vessel Identifiers')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(range(len(vessels)))
    ax1.set_xticklabels(vessels, rotation=45, ha='right')

    # Secondary axis for idle periods
    ax2 = ax1.twinx()
    ax2.plot(vessels, idle_periods, color='red', label='Number of Idle Periods', marker='o')
    ax2.set_ylabel('Number of Idle Periods', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Title and legend
    if port_name is not None:
        plt.title(f'Waiting Times and Idle Periods by Vessel for port {port_name}')
    else:
        plt.title(f'Waiting Times and Idle Periods by Vessel')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Save the plot if save_path is provided
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)

    # Show the plot
    plt.tight_layout()
    plt.show()


# redfeined idle and waiting times for more faster computation 
def analyze_waiting_times(data: pd.DataFrame, ports_df: gpd.GeoDataFrame, 
                          port_name: str = None, center_coords: tuple = None, 
                          offset: float = 1.0, speed_threshold: float = 0.5, 
                          waiting_time_threshold: pd.Timedelta = pd.Timedelta(hours=1), 
                          start_time: str = None, end_time: str = None, 
                          map_save_path: str = None):
    """
    Analyze vessel waiting times in a specific port or region.

    Args:
        data (pd.DataFrame): AIS trajectory data with latitude, longitude, speed, and timestamp.
        ports_df (gpd.GeoDataFrame): GeoDataFrame of ports with polygons.
        port_name (str): Port name to analyze. (Optional)
        center_coords (tuple): Latitude and longitude for the center of a region. (Optional)
        offset (float): Offset for bounding box in degrees. (Default: 1.0)
        speed_threshold (float): Speed below which a vessel is considered idle. (Default: 0.5 knots)
        waiting_time_threshold (timedelta): Time threshold for excessive waiting. (Default: 1 hour)
        start_time (str): Start time for filtering trajectories. (Optional)
        end_time (str): End time for filtering trajectories. (Optional)
        map_save_path (str): Path to save the resulting Folium map. (Optional)

    Returns:
        pd.DataFrame: Summary of vessels exceeding waiting time threshold.
        folium.Map: Folium map showing waiting times and trajectories.
    """
    # Define region based on port or center coordinates
    if port_name:
        region_polygon = ports_df.loc[ports_df['port_name'] == port_name, 'polygon'].iloc[0]
        region_center = region_polygon.centroid.coords[0]
        lon, lat = region_center[0], region_center[1]
    elif center_coords:
        lat, lon = center_coords
    else:
        raise ValueError("Either port_name or center_coords must be provided.")
    
    # Calculate bounding box
    bounding_box = {
        "min_lat": lat - offset,
        "max_lat": lat + offset,
        "min_lon": lon - offset,
        "max_lon": lon + offset
    }
    
    # Filter data within bounding box
    filtered_data = data[
        (data['latitude'] >= bounding_box["min_lat"]) &
        (data['latitude'] <= bounding_box["max_lat"]) &
        (data['longitude'] >= bounding_box["min_lon"]) &
        (data['longitude'] <= bounding_box["max_lon"])
    ]

    # Filter by time range if specified
    if start_time and end_time:
        filtered_data = filtered_data[
            (filtered_data['time_of_position'] >= pd.to_datetime(start_time)) &
            (filtered_data['time_of_position'] <= pd.to_datetime(end_time))
        ]

    if filtered_data.empty:
        print("No data found for the specified region and time range.")
        return pd.DataFrame(), None, None

    # Detect idle states
    filtered_data['is_idle'] = filtered_data['speed'] <= speed_threshold

    # Calculate waiting times
    filtered_data = filtered_data.sort_values(by=['anonymized_mmsi', 'time_of_position'])
    filtered_data['time_diff'] = filtered_data.groupby('anonymized_mmsi')['time_of_position'].diff()

    filtered_data['cumulative_wait'] = filtered_data.groupby('anonymized_mmsi')['time_diff'].cumsum()
    filtered_data['waiting_flag'] = (
        filtered_data['is_idle'] & 
        (filtered_data['cumulative_wait'] >= waiting_time_threshold)
    )

    # Aggregate waiting times
    waiting_summary = filtered_data[filtered_data['waiting_flag']].groupby('anonymized_mmsi').agg(
        total_waiting_time=('cumulative_wait', 'max'),
        num_idle_periods=('waiting_flag', 'sum')
    ).reset_index()

    # Create Folium map
    m = folium.Map(location=[lat, lon], zoom_start=5)
    folium.Rectangle(
        bounds=[
            (bounding_box["min_lat"], bounding_box["min_lon"]),
            (bounding_box["max_lat"], bounding_box["max_lon"])
        ],
        color='red',
        fill=True,
        fill_opacity=0.2,
        popup="Analysis Region"
    ).add_to(m)

    # Add idle trajectories
    idle_data = filtered_data[filtered_data['is_idle']]
    for _, row in idle_data.iterrows():
        folium.CircleMarker(
            location=(row['latitude'], row['longitude']),
            radius=2,
            color='blue',
            fill=True,
            fill_opacity=0.7,
            popup=f"Idle at: {row['time_of_position']}"
        ).add_to(m)

    # Save map if path is specified
    if map_save_path:
        m.save(map_save_path)
        
    visualize_waiting_times(waiting_summary, port_name)
    
    return waiting_summary, idle_data, m


def analyze_port_congestion(data, number_of_days_thrsh = 5, selected_ports=None):
    """
    Analyze port congestion based on waiting times and thresholds.

    Parameters:
        data (pd.DataFrame): DataFrame containing 'port_name', 'total_waiting_time', and 'distinct_vessels'.
        number_of_days_thrsh (float): Constant multiplier to determine thresholds (default: 5 days).
        selected_ports (list of str): List of ports to include in the analysis (default: None, include all ports).

    Returns:
        pd.DataFrame: Congestion analysis results.
        matplotlib.figure.Figure: Visualization of congestion levels.
    """
    # Convert total_waiting_time to a total number of days (in case it's in timedelta format)
    # data['total_waiting_time_days'] = data['total_waiting_time'].apply(lambda x: x.total_seconds() / (60 * 60 * 24))
    # data['total_waiting_time_days'] = data['wainting'].apply(lambda x: x.total_seconds() / (60 * 60 * 24))

    # Filter by selected ports if provided
    if selected_ports is not None:
        data = data[data['port_name'].isin(selected_ports)]
    
    # Count distinct vessels per port (in case 'num_idle_periods' or a similar metric exists)
    distinct_vessels = data.groupby('port_name')['anonymized_mmsi'].nunique()
    
    # Merge distinct vessel counts back to data
    data = data.merge(distinct_vessels, on='port_name', suffixes=('', '_distinct_vessels'))
    
    # Calculate thresholds
    data["threshold"] = data["anonymized_mmsi_distinct_vessels"] * number_of_days_thrsh
    # Determine congestion status
    data["is_congested"] = data["waiting_time_days"] > data["threshold"]
    
    # Summarize per port for visualization
    port_summary = data.groupby('port_name').agg({
        'waiting_time_days': 'max',
        'anonymized_mmsi_distinct_vessels': 'max',
        'threshold': 'max',
        'is_congested': 'max'
    }).reset_index()

    # Visualization
    plt.figure(figsize=(10, 6))
    bars = plt.bar(port_summary['port_name'], port_summary['waiting_time_days'], color='blue', alpha=0.7, label="Total Waiting Time (days)")
    thresholds = plt.plot(port_summary['port_name'], port_summary['threshold'], color='red', marker='o', label="Threshold (days)")
    
    # Highlight congested ports in red
    for i, bar in enumerate(bars):
        if port_summary["is_congested"].iloc[i]:
            bar.set_color('blue')
    
    plt.xlabel("Port Names")
    plt.ylabel("Waiting Time (days)")
    plt.title("Port Congestion Analysis")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    return port_summary, plt



# --------------------- predictions -----------------------------

def prepare_traffic_data(
    raw_data: pd.DataFrame,
    port_gdf: pd.DataFrame,
    port_name: str = None,
    center_coords: tuple = None,
    offset: float = 1.0,
    time_interval: str = "daily",
) -> pd.DataFrame:
    """
    Prepare traffic data for analysis by filtering trajectories based on bounding box.

    Parameters:
        raw_data (pd.DataFrame): AIS trajectory data.
        port_gdf (pd.DataFrame): DataFrame containing port polygons and metadata.
        port_name (str, optional): Name of the port to analyze. Defaults to None.
        center_coords (tuple, optional): (latitude, longitude) for custom region. Defaults to None.
        offset (float, optional): Bounding box size in degrees. Defaults to 1.0.
        time_interval (str, optional): Aggregation interval ('daily', 'weekly', 'monthly'). Defaults to 'daily'.

    Returns:
        pd.DataFrame: Aggregated data for the specified region or port.
    """
    # Determine bounding box
    if port_name:
        # Find port polygon center
        port_row = port_gdf[port_gdf["port_name"] == port_name]
        if port_row.empty:
            raise ValueError(f"Port '{port_name}' not found in port_gdf.")
        center_coords = (
            port_row["polygon"].centroid.y.values[0],
            port_row["polygon"].centroid.x.values[0],
        )
    
    if not center_coords:
        raise ValueError("Either port_name or center_coords must be provided.")

    # Apply bounding box logic
    lat, lon = center_coords
    bounding_box = {
        "min_lat": lat - offset,
        "max_lat": lat + offset,
        "min_lon": lon - offset,
        "max_lon": lon + offset,
    }

    filtered_data = raw_data[
        (raw_data["latitude"] >= bounding_box["min_lat"]) &
        (raw_data["latitude"] <= bounding_box["max_lat"]) &
        (raw_data["longitude"] >= bounding_box["min_lon"]) &
        (raw_data["longitude"] <= bounding_box["max_lon"])
    ]

    if filtered_data.empty:
        print("No data found in the specified region.")
        return pd.DataFrame()

    # Convert timestamps to datetime if not already
    filtered_data["time_of_position"] = pd.to_datetime(filtered_data["time_of_position"])

    # Aggregate data
    if time_interval == "daily":
        filtered_data["time_period"] = filtered_data["time_of_position"].dt.date
    elif time_interval == "weekly":
        filtered_data["time_period"] = filtered_data["time_of_position"].dt.to_period("W")
    elif time_interval == "monthly":
        filtered_data["time_period"] = filtered_data["time_of_position"].dt.to_period("M")
    else:
        raise ValueError("Invalid time_interval. Choose 'daily', 'weekly', or 'monthly'.")

    aggregated_data = filtered_data.groupby("time_period").agg({
        "anonymized_mmsi": "nunique",  # Number of unique vessels
        "speed": "mean",              # Average speed
        "time_of_position": "count"          # Number of records
    }).reset_index()

    aggregated_data.rename(
        columns={
            "anonymized_mmsi": "vessel_count",
            "speed": "average_speed",
            "time_of_position": "trajectory_count",
        },
        inplace=True,
    )

    return aggregated_data


def forecast_traffic_ARIMA(data, metric='vessel_count', forecast_period=30, granularity='daily'):
    """
    Forecast vessel traffic using ARIMA.

    Parameters:
        data (pd.DataFrame): Time-series data with a 'time_period' column and the metric to forecast.
        metric (str): Metric to forecast ('vessel_count', 'trajectory_count', or 'average_speed').
        forecast_period (int): Number of periods to forecast (default: 30).
        granularity (str): Time granularity ('daily', 'weekly', or 'monthly', default: 'daily').

    Returns:
        pd.DataFrame: DataFrame with observed and predicted values.
        matplotlib.figure.Figure: Forecast visualization.
    """
    # Ensure time_period is a datetime index
    data['time_period'] = pd.to_datetime(data['time_period'])
    data.set_index('time_period', inplace=True)

    # Resample data based on granularity
    if granularity == 'weekly':
        data = data.resample('W').mean()
    elif granularity == 'monthly':
        data = data.resample('M').mean()
    elif granularity == "daily":
        data = data.resample('D').sum()

    # Train ARIMA model
    y = data[metric].fillna(0)
    model = ARIMA(y, order=(1, 0, 1))  # Simple ARIMA model; can be tuned
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=forecast_period)
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_period, freq='D')

    # Create results DataFrame
    forecast_df = pd.DataFrame({metric: forecast}, index=forecast_index)
    result_df = pd.concat([data[[metric]], forecast_df], axis=0)

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(result_df.index, result_df[metric], label='Observed', color='blue')
    plt.plot(forecast_df.index, forecast_df[metric], label='Forecast', color='orange', linestyle='--')
    plt.title(f"Forecast for {metric} ({granularity})")
    plt.xlabel("Time")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid()

    return result_df, plt


# Prepare training and testing data for XGBoost
def preprocess_xgboost_data(df):
    df = df.dropna()
    
    X = df
    y = df['vessel_count'].shift(-1).dropna()
    X = X[:-1]  # Drop the last row due to shifting
    return X, y


def train_xgboost_model(X_train, y_train, X_test, y_test):
    model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                             max_depth=5, alpha=10, n_estimators=100)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    return predictions


def visualize_xgboost_predictions(y_test, xgboost_predictions):
    """
    Visualizes the XGBoost predictions against the actual values.
    
    Args:
    - y_test: Actual test values.
    - xgboost_predictions: Predicted values from the XGBoost model.
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(y_test, label='Actual', color='blue')
    plt.plot(xgboost_predictions, label='XGBoost Predictions', color='green')
    
    plt.title('XGBoost Predictions vs Actual')
    plt.xlabel('Time Period')
    plt.ylabel('Vessel Count')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
