import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely import wkt
from tqdm import tqdm
import os

import zipfile
import io
import numpy as np

def get_seongsu_bus():
    # Initialize the coordinate transformer
    transformer = Transformer.from_crs("epsg:4326", "epsg:5179")

    # Function to transform latitude and longitude to EPSG:5179
    def transform_coordinates(lat, lon):
        x, y = transformer.transform(lat, lon)
        return x, y

    # Load the master dataset and rename columns
    master_df = pd.read_csv('datasets/bike/raw/station_master.csv', encoding='cp949')
    new_column_names_master = ['st_id', 'addr1', 'addr2', 'latitude', 'longtitude']
    master_df.columns = new_column_names_master

    # Transform latitude and longitude to EPSG:5179
    master_df['y'], master_df['x'] = zip(*master_df.apply(lambda row: transform_coordinates(row['latitude'], row['longtitude']), axis=1))

    # Load the polygon dataset and rename columns
    polygon_df = pd.read_csv('datasets/polygon/sungsoo_match_60_polygon.csv', encoding='ISO-8859-1')
    new_column_names_polygon = ['cell_id_1', 'cell_id_2', 'cell_info']
    polygon_df.columns = new_column_names_polygon

    # Function to find the polygon containing the point and return the corresponding cell_id_1 and cell_id_2
    def find_polygon_mapping(x, y):
        point = Point(x, y)
        for _, polygon_row in polygon_df.iterrows():
            polygon = wkt.loads(polygon_row['cell_info'])  # Load WKT string into polygon
            if polygon.contains(point):
                return polygon_row['cell_id_1'], polygon_row['cell_id_2']  # Return cell_id_1 and cell_id_2 if point is in the polygon
        return None, None  # Return None if no polygon contains the point

    # Apply the function dynamically for each point in master_df, with tqdm progress bar
    tqdm.pandas()  # Activate tqdm for pandas

    # Use progress_apply to map the cell_id_1 and cell_id_2
    master_df[['cell_id_1', 'cell_id_2']] = master_df.progress_apply(
        lambda row: find_polygon_mapping(row['x'], row['y']), axis=1, result_type='expand'
    )

    # Print the result
    #print("\nDataFrame with 'cell_id_1' and 'cell_id_2' columns:")
    #print(master_df.head())

    master_df = master_df.loc[(master_df['cell_id_1'].notna()) | (master_df['cell_id_2'].notna())]
    master_df = master_df.drop(['cell_id_1', 'cell_id_2'], axis = 1)
    master_df.to_csv('datasets/bike/seongsu/bike_master_added.csv', index=False, encoding='cp949')

#get_seongsu_bus()
def calculate_return_hour(row):
    # Extract hour and minute from rent_hour
    rent_hour = row['rent_hour']  # e.g., 630
    trip_minute = row['trip_minute']

    # Extract hours and minutes from rent_hour
    hour = rent_hour // 100  # Get the hour part
    minute = rent_hour % 100  # Get the minute part

    # Add trip minutes to the minutes
    total_minutes = minute + trip_minute

    # Calculate new hour and minute
    new_hour = hour + total_minutes // 60
    new_minute = total_minutes % 60

    # Round minutes to the nearest 5 minutes
    new_minute = (new_minute // 5) * 5

    # Adjust for hour overflow if minutes become 60 or more
    if new_minute >= 60:
        new_minute -= 60
        new_hour += 1

    # Ensure hour is in HH format (e.g., 1 becomes 010, 10 becomes 100)
    return_hour = new_hour * 100 + new_minute
    return return_hour


def extract_bike_info():
    # Load station master data
    station_master = pd.read_csv('datasets/bike/seongsu/bike_master_added.csv', encoding='cp949')
    valid_stations = station_master['st_id'].tolist()

    # Define the output combined CSV file
    combined_csv_file = 'datasets/bike/seongsu/seongsu_combined_bike_202301_202406.csv'

    all_data = []  # List to store all dataframes

    # Loop through multiple zip files from 202301 to 202406
    for month in tqdm(range(202401, 202407), desc='Processing CSV files'):
        zip_file = f'datasets/bike/raw/tpss_bcycl_od_statnhm_{month}.zip'
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as z:
                for csv_file in z.namelist():
                    if csv_file.endswith('.csv'):
                        # Read each CSV file inside the zip
                        with z.open(csv_file) as f:
                            
                            trip_data = pd.read_csv(f, encoding = 'cp949')

                            if '집계_기준' in trip_data.columns:
                                trip_data = trip_data.loc[trip_data['집계_기준'] == '출발시간'] 

                            trip_data = trip_data.loc[:, ~trip_data.columns.str.contains('^Unnamed|^$', regex=True)]

                            # Define new column names for the trip data
                            required_columns = ['집계_기준', '시작_대여소명', '종료_대여소명', '']

                            # Identify which required columns exist in trip_data
                            existing_columns = [col for col in required_columns if col in trip_data.columns]

                            # Drop the existing columns from trip_data
                            trip_data = trip_data.drop(columns=existing_columns)

                            #print('trip_data.columns: ', trip_data.columns)

                            new_column_names_trip = ['date', 'rent_hour', 'rent_st_id', 'return_st_id', 'trip_count', 'trip_minute', 'trip_distance']
                            
                            try:
                                trip_data.columns = new_column_names_trip
                            except ValueError as e:
                                print(f"ValueError: {e}")
                                print('filename: ', f)
                                print('trip_data.columns: ', trip_data.columns)
                                print(f"Expected {len(new_column_names_trip)} elements, but got {trip_data.shape[1]} elements.")
                                # Optionally, you can handle the mismatch here
                            

                            trip_data['return_hour'] = trip_data.apply(calculate_return_hour, axis=1)
                            
                            # Group by date, rent_st_id, return_st_id and calculate sum or mean as needed
                            #grouped_data = trip_data.groupby(['date', 'rent_st_id', 'return_st_id'], as_index=False).agg({
                            #    'trip_count': 'sum', 
                            #    'trip_minute': 'mean', 
                            #    'trip_distance': 'mean'
                            #})
                            
                            # Create the 'seongsu_in' column: 1 if return_st_id is in valid_stations, 0 if rent_st_id is in valid_stations
                            trip_data['seongsu_in'] = np.where(trip_data['return_st_id'].isin(valid_stations), 1, np.nan)
                            
                            # Filter rows where either rent_st_id or return_st_id is in the valid stations
                            filtered_data = trip_data[trip_data['seongsu_in'].notna()]

                            # Append filtered data to the list
                            all_data.append(filtered_data)
        except FileNotFoundError:
            print(f"File not found: {zip_file}")
            continue

    # Concatenate all dataframes into one
    combined_data = pd.concat(all_data, ignore_index=True)

    # Sort combined_data by 'date' in ascending order
    combined_data = combined_data.sort_values(by='date', ascending=True)

    # Save the combined data into a single CSV file
    combined_data.to_csv(combined_csv_file, index=False, encoding = 'cp949')
    
    print(f"Combined data saved to {combined_csv_file}")

extract_bike_info()