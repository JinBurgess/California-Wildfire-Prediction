from common_imports import *

from lib.data_processing import DataProcessor
processor = DataProcessor()
#______________________________________________________________________________________________________________________________
#Load data
dir_path_merged_cities = "weather_data/Merged_City"
weather_df = processor.merge_files(dir_path_merged_cities)
fire_df= pd.read_csv("wildfire_data/wildfire_with_severity.csv")

# Merge weather and wildfire data
merged_df = processor.merge_weather_fire(weather_df, fire_df, "weather_data/city_location.csv")

# The merged DataFrame is also saved as 'lookup.csv'
print(merged_df.head())