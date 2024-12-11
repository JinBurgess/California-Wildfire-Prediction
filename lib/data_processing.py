from common_imports import *

class DataProcessor:
    def __init__(self):
        pass

    def merge_files(self, dir_path):
        """
        Merges all CSV files in the specified directory into a single DataFrame.
        """
        files = [
            pd.read_csv(os.path.join(dir_path, filename)) 
            for filename in os.listdir(dir_path) 
            if filename.endswith('.csv')
        ]
        merged_df = pd.concat(files, ignore_index=True)
        return merged_df

    def preprocess_df(self, df, weather_features):
        """
        Preprocesses a weather DataFrame by normalizing features, engineering columns, and filling missing values.
        """
        df['datetime'] = pd.to_datetime(df['datetime'])  # Convert 'datetime' column to datetime
        df.sort_values(by=['name', 'datetime'], inplace=True)

        # Scale numeric features
        scaler = StandardScaler()
        df[weather_features] = scaler.fit_transform(df[weather_features])

        df['FireOccurred'] = df['Started'].notna().astype(int)
        df = df.drop(columns=['windgust', 'Lat', 'Long', 'Duration', 'Started', 'OptimizedSeverityScore', "closest_city"])

        df.fillna(0, inplace=True)
        return df

    def merge_weather_fire(self, weather_df, fire_severity, city_loc_path):
        """
        Merges weather data with wildfire severity data based on closest city and date.

        Parameters:
            weather_df (pd.DataFrame): The weather data.
            fire_severity (pd.DataFrame): The wildfire severity data.
            city_loc_path (str): Path to the city location CSV file.

        Returns:
            pd.DataFrame: Merged DataFrame of weather and wildfire data.
        """
        # Filter relevant columns from the weather data
        weather_df = weather_df[['name', 'datetime', 'temp', 'dew', 'precip', 'precipcover', 
                                 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir', 
                                 'sealevelpressure', 'cloudcover', 'solarradiation']]
        
        # Load city location data
        city_loc = pd.read_csv(city_loc_path)
        merge_df = pd.merge(weather_df, city_loc, on='name', how='inner')

        # Calculate distances between wildfires and cities
        wildfire_coords = fire_severity[['Latitude', 'Longitude']].to_numpy()
        wildfire_coords_rad = np.radians(wildfire_coords)  # Convert to radians

        city_loc['Lat'] = pd.to_numeric(city_loc['Lat'], errors='coerce')
        city_loc['Long'] = pd.to_numeric(city_loc['Long'], errors='coerce')
        city_coords = city_loc[['Lat', 'Long']].to_numpy()
        city_coords_rad = np.radians(city_coords)  # Convert to radians

        distances = cdist(wildfire_coords_rad, city_coords_rad) * 6371 # distances in km 

        # ensure cloest city is within 170 km 
        closest_city_idx = np.argmin(distances, axis=1)

        closest_city_distances = distances[np.arange(len(distances)), closest_city_idx]
        closest_cities = np.where(closest_city_distances <= 160, city_loc.iloc[closest_city_idx]['name'].values, np.nan)

        # Add closest city information to wildfire data
        fire_severity['closest_city'] = closest_cities

        # Convert datetime columns to dates for merging
        merge_df['datetime'] = pd.to_datetime(merge_df['datetime'], format='mixed', errors='coerce').dt.date
        fire_severity['Started'] = pd.to_datetime(fire_severity['Started'], format='mixed', errors='coerce').dt.date

        # Merge weather and wildfire data
        merged_df = pd.merge(
            merge_df,
            fire_severity[['Started', 'Duration', 'OptimizedSeverityScore', 
                           'OptimizedSeverityScore_Log', 'closest_city']],
            left_on=['datetime', 'name'],
            right_on=['Started', 'closest_city'],
            how='left'
        )

        # Save the merged data to a CSV file
        merged_df.to_csv("lookup.csv", index=False)
        return merged_df
