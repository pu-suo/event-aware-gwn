import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance in kilometers between two points
    on the earth (specified in decimal degrees).
    """
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371  # Radius of earth in kilometers.
    return c * r

def main(args):
    """
    Main function to generate the dynamic event feature tensor.
    This script pre-computes a tensor of shape (T, N, F) where T is the number of
    timestamps, N is the number of sensors, and F is the number of event features.
    """
    print("Loading data...")
    try:
        meta_df = pd.read_csv(args.meta_data_path)
        sensor_coords = {row['sensor_id']: (row['Latitude'], row['Longitude']) for _, row in meta_df.iterrows()}
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading metadata from {args.meta_data_path}: {e}")
        return

    sensor_ids = sorted(list(sensor_coords.keys()))
    num_sensors = len(sensor_ids)

    try:
        events_df = pd.read_csv(args.events_data_path)
        events_df['event_start'] = pd.to_datetime(events_df['event_start'])
        events_df['event_end'] = pd.to_datetime(events_df['event_end'])
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading events data from {args.events_data_path}: {e}")
        return

    max_attendance = events_df['attendance'].max()
    events_df['normalized_attendance'] = events_df['attendance'] / max_attendance if max_attendance > 0 else 0

    try:
        traffic_df = pd.read_csv(args.traffic_data_path, index_col=0)
        traffic_df.index = pd.to_datetime(traffic_df.index)
        timestamps = sorted(traffic_df.index.unique())
    except FileNotFoundError:
        print(f"Error: Traffic data file not found at {args.traffic_data_path}")
        return
    
    num_timestamps = len(timestamps)

    print(f"Found {num_sensors} sensors and {num_timestamps} unique timestamps.")
    print(f"Processing {len(events_df)} events.")

    feature_names = [
        'spatial_relevance_score', 'time_to_start', 'time_since_end',
        'event_active_flag', 'normalized_attendance', 'is_Basketball', 'is_Hockey'
    ]
    num_features = len(feature_names)

    event_features = np.zeros((num_timestamps, num_sensors, num_features), dtype=np.float32)

    print("Generating event features for each timestamp and sensor...")
    for t_idx, ts in enumerate(tqdm(timestamps, desc="Processing Timestamps")):
        
        active_events = events_df[(events_df['event_start'] <= ts) & (events_df['event_end'] >= ts)]

        if active_events.empty:
            continue

        for s_idx, sensor_id in enumerate(sensor_ids):
            sensor_lat, sensor_lon = sensor_coords[sensor_id]

            distances = [haversine_distance(sensor_lat, sensor_lon, row['latitude'], row['longitude']) for _, row in active_events.iterrows()]
            
            if not distances:
                continue

            nearest_event_idx = np.argmin(distances)
            nearest_event = active_events.iloc[nearest_event_idx]
            distance = distances[nearest_event_idx]

            # --- Feature Calculation ---
            spatial_relevance = np.exp(-(distance**2) / (args.spatial_radius**2))
            
            time_diff_start_hours = (nearest_event['event_start'] - ts).total_seconds() / 3600
            time_to_start = time_diff_start_hours / args.temporal_normalization_hours

            time_diff_end_hours = (ts - nearest_event['event_end']).total_seconds() / 3600
            time_since_end = max(0, time_diff_end_hours / args.temporal_normalization_hours)
            
            event_active_flag = 1.0
            norm_attendance = nearest_event['normalized_attendance']
            
            is_basketball = 1.0 if nearest_event['category'] == 'Basketball' else 0.0
            is_hockey = 1.0 if nearest_event['category'] == 'Hockey' else 0.0

            event_features[t_idx, s_idx, :] = [
                spatial_relevance, time_to_start, time_since_end,
                event_active_flag, norm_attendance, is_basketball, is_hockey
            ]

    print("Feature generation complete.")
    print(f"Final tensor shape: {event_features.shape}")

    print(f"Saving features to {args.output_path}...")
    np.savez_compressed(
        args.output_path,
        features=event_features,
        sensor_ids=np.array(sensor_ids),
        feature_names=np.array(feature_names),
        timestamps=np.array(timestamps)
    )
    print("Save complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dynamic event features for traffic forecasting.")
    parser.add_argument('--traffic_data_path', type=str, default='../data/PEMS-BAY.csv',
                        help='Path to the PEMS-BAY traffic data CSV file.')
    parser.add_argument('--meta_data_path', type=str, default='../data/PEMS-BAY-META.csv',
                        help='Path to the PEMS-BAY sensor metadata CSV file.')
    parser.add_argument('--events_data_path', type=str, default='../data/pems_events.csv',
                        help='Path to the structured events data CSV file.')
    parser.add_argument('--output_path', type=str, default='../data/event_features.npz',
                        help='Path to save the generated event feature tensor.')
    parser.add_argument('--spatial_radius', type=float, default=10.0,
                        help='The spatial radius of influence for an event in kilometers (hyperparameter sigma).')
    parser.add_argument('--temporal_normalization_hours', type=float, default=6.0,
                        help='Normalization constant for temporal features in hours.')

    args = parser.parse_args()
    main(args)

