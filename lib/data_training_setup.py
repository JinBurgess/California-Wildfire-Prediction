from common_imports import *

class DataSetup:
    def __init__(self, sequence_length, weather_features):
        """
        Initializes the data setup class.

        Parameters:
            sequence_length (int): Length of the weather data sequence.
            weather_features (list): List of weather features to be used.
        """
        self.sequence_length = sequence_length
        self.weather_features = weather_features

    def tt_split(self, locations, df, by="name"):
        """
        Splits data into time series samples for each location (city or county).

        Parameters:
            locations (list): List of unique locations (cities or counties).
            df (pd.DataFrame): DataFrame containing the data.
            by (str): Specifies grouping by 'city' or 'county'.

        Returns:
            pd.DataFrame: DataFrame with time series samples for each location.
        """
        all_X = []
        for loc in locations:
            loc_df = df[df[by] == loc]
            for i in range(len(loc_df) - self.sequence_length):
                weather_data = loc_df[self.weather_features].iloc[i:i + self.sequence_length].values
                fire_label = loc_df['FireOccurred'].iloc[i + self.sequence_length]
                severity_label = (
                    loc_df['OptimizedSeverityScore_Log'].iloc[i + self.sequence_length]
                    if fire_label == 1 else 0
                )
                associated_date = loc_df['datetime'].iloc[i + self.sequence_length]
                all_X.append({
                    by: loc,
                    'datetime': associated_date,
                    'weather_data': weather_data,
                    'y_fire': fire_label,
                    'y_severity': severity_label
                })

        return pd.DataFrame(all_X)

    def train_val_test_split(self, train_df, test_df, locations, split_by="name"):
        """
        Splits the data into training, validation, and test sets.

        Parameters:
            train_df (pd.DataFrame): Training data.
            test_df (pd.DataFrame): Test data.
            locations (list): List of unique locations (cities or counties).
            grouped (bool): Whether to use grouped splitting. Default is False.
            split_by (str): Splitting criterion, either 'city' or 'county'.

        Returns:
            tuple: DataFrames for train, validation, and test sets.
        """
        # Separate validation and test sets
        validation = test_df[test_df['datetime'] < '2019-01-01']
        test = test_df[test_df['datetime'] >= '2019-01-01']

        # Perform splitting
        X_train = self.tt_split(locations, train_df, by=split_by)
        X_val = self.tt_split(locations, validation, by=split_by)
        X_test = self.tt_split(locations, test, by=split_by)

        return X_train, X_val, X_test

    def setup_x_and_y(self, df):
        """
        Prepares feature and label arrays for training and validation.

        Parameters:
            train (pd.DataFrame): Training data with weather and label columns.
            val (pd.DataFrame): Validation data with weather and label columns.

        Returns:
            tuple: Numpy arrays for features and labels (fire occurrence and severity).
        """
        weather_features = np.stack(np.array(df['weather_data'].values))

        y_train_fire = np.stack(df['y_fire'].values)
        y_train_fire = y_train_fire.astype(int)
        y_train_severity = np.stack(df['y_severity'].values)

        return weather_features, y_train_fire, y_train_severity
