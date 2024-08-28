# This code is used to prepare the data files needed for the model, if there are any technical problems, please contact: gongyaofa0211@g.ucla.edu

# 0. Loading python libraries

from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
from geopandas.tools import sjoin
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

# 1. Match the road segment where the work zone occurs(It can also be used to process incident data)

# Load data
# Loading road network data
road_network_path = 'raw data/TMC_Identification.geojson'
road_network_gdf = gpd.read_file(road_network_path)

# Loading work zone data
incident_path = 'raw data/MD_workzones_adjust.csv'
incident_gdf = gpd.read_file(incident_path)

# Convert work zone data to GeoDataFrame
incident_gdf['geometry'] = incident_gdf.apply(
    lambda row: Point(float(row['start_longitude']), float(row['start_latitude'])), axis=1)
work_zone_gdf = gpd.GeoDataFrame(incident_gdf, geometry='geometry')
# Ensure that both GeoDataFrames use the same coordinate reference system
work_zone_gdf.crs = road_network_gdf.crs

# Define the function that calculates the distance
def calculate_distance(road, incident):
    return road.geometry.distance(incident.geometry)

# Define a function to find the nearest road segment
def process_incident(workzone):

    # Iterate over the entire data
    # Filtering of road sections with the same "direction" attribute
    filtered_roads = road_network_gdf[road_network_gdf['direction'] == workzone['direction']]
    if filtered_roads.empty:
        return None

    # Calculate the distance of these sections from the work zone point
    filtered_roads['distance'] = filtered_roads.geometry.distance(workzone.geometry)

    # Find the nearest roadway and keep the following labels
    nearest_road = filtered_roads.nsmallest(1, 'distance').iloc[0]

    # Here you can set the columns you want to keep, and since we're splitting the two pieces of data (the road network data and the work zone data) to form a new piece of data, we need to define which variables we need to make them appear in the new data
    # In the string on the left we can define the name of the column we want to leave behind, and the string on the right represents which dataset this data comes from (road network data or work zone data)
    return {
        'workzone_id': workzone['event_id'],
        'tmc': nearest_road['tmc'],
        'direction': workzone['direction'],
        'road_inc': workzone['road_name_1'],
        'road_tmc': nearest_road['road'],
        'start_time': workzone['start_date'],
        'end_time': workzone['end_date'],
        'description': workzone['description'],
        # 'weather': workzone['weather'],
    }

# Optimize program speed
executor = ThreadPoolExecutor(max_workers=20)  # Adjustment to system resources(eg. max_workers=10 or 20)
futures = [executor.submit(process_incident, incident) for _, incident in work_zone_gdf.iterrows()]
results_list = [future.result() for future in futures if future.result() is not None]

# Creating a list of results
matched_results = pd.DataFrame(results_list)

# Filter out rows where road_tmc and road are inconsistent
mismatched_rows = matched_results[matched_results['road_tmc'] != matched_results['road_inc']]

# Save these lines to a separate file
mismatched_rows.to_csv('raw data/MD_incidents_mis.csv',
                       index=False)

# Delete these rows from the original data
matched_results = matched_results[matched_results['road_tmc'] == matched_results['road_inc']]

# Save the remaining matches
matched_results.to_csv('raw data/MD_incidents_mat.csv',
                       index=False)

# 2. Handle incident data in the same way as above, match the location of occurrence for each incident data
# Note: The above code can be used to match the incident data in the same way, we need to match the incident data(MD_incidents_adjust.csv) with the same tmc section number in order to merge the data later.

# 3. Match the work zone data with speed data(If the speed data is too large, consider splitting the entire speed data first, and then merging the whole datasets after doing them one by one.)

# Set path to the loaded file
speed_data = pd.read_csv('raw data/speed_sample_2019_10_01_103P10392.csv')

# Convert a date string to a datetime object
matched_results['start_time'] = pd.to_datetime(matched_results['start_time'])
matched_results['end_time'] = pd.to_datetime(matched_results['end_time'])
speed_data['measurement_tstamp'] = pd.to_datetime(speed_data['measurement_tstamp'])

# Set the duration of the observation before and after the start of the work zone, which can be flexibly adjusted here. (Example: 12 hours before the start of the event until 2 hours after the start of the event.)
matched_results['time_start_record'] = matched_results['start_time'] - timedelta(hours=12)
matched_results['time_end_record'] = matched_results['end_time'] + timedelta(hours=2)

# Select the columns to merge(You can make your own adjustments here to remove columns that don't need to be merged.)
columns_to_merge = ['tmc', 'workzone_id', 'start_time', 'end_time', 'Vehicles Involved', 'direction', 'Max Lanes Closed', 'Standardized Type']

# Selection of specific columns from road segment information data
matched_results_selected = matched_results[columns_to_merge]

# Merge data using the merge function
combined_data = pd.merge(left=speed_data, right=matched_results_selected,
                         left_on='tmc_code', right_on='n_tmc',
                         how='inner')

# Further screening of merged data to meet conditions such as date range
combined_data = combined_data[(combined_data['measurement_tstamp'] >= combined_data['start_time']) &
                              (combined_data['measurement_tstamp'] <= combined_data['end_time'])]

# Save the merged data to a new CSV file
# output_file_path = 'H:\DALL_CODE\DAII\data/2016_2019\incidents_ritis\split_merge_with_speed/merged_data.csv'
# combined_data.to_csv(output_file_path, index=False)

# 4. Match the work zone data with incident data

# Load the matched incident data
df_mat = pd.read_csv('raw data/MD_incidents_mat.csv')

# Extract the contents of '[]' using regular expressions
df_mat['description'] = df_mat['description'].str.extract('\[(.*?)\]', expand=False)

# Filter rows containing 'Collision'
df_collision_mat = df_mat[df_mat['description'].str.contains('Collision', na=False)]

# Convert start_time and end_time columns to datetime objects
df_collision_mat['start_time'] = pd.to_datetime(df_collision_mat['start_time'])
df_collision_mat['end_time'] = pd.to_datetime(df_collision_mat['end_time'])

# Round the time back to a multiple of 15 minutes
df_collision_mat['start_time'] = df_collision_mat['start_time'].dt.ceil('15T')
df_collision_mat['end_time'] = df_collision_mat['end_time'].dt.ceil('15T')

# Construct lists to hold new rows
new_rows = []

# Iterate over each row, generate time series, create new rows
for _, row in df_collision_mat.iterrows():
    
    # Generate time series at 15-minute intervals
    time_range = pd.date_range(start=row['start_time'], end=row['end_time'], freq='15T')
    
    # For each point in the time series, create a new row
    for current_time in time_range:
        new_row = row.to_dict()
        new_row['current_time_record'] = current_time
        new_rows.append(new_row)

# Create a new DataFrame
new_df = pd.DataFrame(new_rows)

# Delete unwanted columns
new_df.drop(['direction', 'road_inc', 'road_tmc', 'start_time', 'end_time'], axis=1, inplace=True)

# Consolidation of data
merged_df = pd.merge(left=combined_data, right=new_df, how='left', left_on=['tmc_code', 'measurement_tstamp'], right_on=['n_tmc', 'current_time_record'])

# Save data
# merged_df.to_csv("path_to_save.csv",index=Flase)

# 5. Fill in the week name and time

# Add the week column and populate it with the day of the week based on the measurement_tstamp column
merged_df['week'] = merged_df['measurement_tstamp'].dt.day_name()

# Add the time column and extract only the hours, minutes, and seconds portion of the measurement_tstamp column
merged_df['time'] = merged_df['measurement_tstamp'].dt.time

# 6. Filling of AADT

# Importing AADT data
AADT_data = pd.read_csv("raw data/MD_AADT_2019_adjust.csv", low_memory=False)

# Define a function to handle unstandardized AADT data for better merging.
def format_time(time_str):
    # If the length of the time string is 5 (i.e., HH:MM format), add ":00" at the end
    # Here is because in the original AADT, there is no time column, we need to generate the corresponding time according to the "time_bin" column, if the generated time is not "HH: MM: SS" format, you need to use this function to adjust the format.
    if len(time_str) == 5:
        time_str += ":00"
    return pd.to_datetime(time_str, format='%H:%M:%S').time()

# Add two new columns to speed_data and initialise to None or 0
merged_df['volume'] = None

# Convert time columns to a uniform datetime format
merged_df['time'] = pd.to_datetime(merged_df['time'], format='%H:%M:%S').dt.time

# Apply format_time function
AADT_data['time'] = AADT_data['time'].apply(format_time)

# Merge data using the merge() function
merged_data_with_AADT = pd.merge(speed_data, AADT_data, left_on=['tmc_code', 'week', 'time'],
                       right_on=['tmc', 'day_of_week', 'time'], how='left')

# update value
merged_data_with_AADT['volume'] = merged_data_with_AADT['volume_y']

# Delete the extra column
columns_to_drop = ['Unnamed: 0_x', 'Unnamed: 0_y', 'time', 'tmc', 'day_of_week', 'time_bin', 'percent_commercial', 'volume_y', 'volume_x']
merged_data_with_AADT.drop(columns=columns_to_drop, inplace=True)