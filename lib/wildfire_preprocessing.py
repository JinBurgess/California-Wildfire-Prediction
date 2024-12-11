from common_imports import *

class WildfirePreprocessor:
    def __init__(self):
        self.best_weights = None
        self.best_estimator = None

    def filter_columns(self, df):
        """
        Drops unnecessary columns from the wildfire DataFrame.
        """
        columns_to_drop = [
            'PercentContained', 'CanonicalUrl', 'ConditionStatement',
            'SearchDescription', 'SearchKeywords', 'Featured', 'Final', 'Status'
        ]
        df_filtered = df.drop(columns=columns_to_drop)
        return df_filtered

    def process_datetime_columns(self, df):
        """
        Processes and calculates date and duration-related features.
        """
        df['Started'] = pd.to_datetime(df['Started'], format="ISO8601", errors='coerce').dt.tz_localize(None)
        df['Extinguished'] = pd.to_datetime(df['Extinguished'], format="ISO8601", errors='coerce').dt.tz_localize(None)
        df['Duration'] = (df['Extinguished'] - df['Started']).dt.days.fillna(0)
        df['Duration'] = df['Duration'].clip(upper = 100)
        return df

    def clean_and_convert(self, df):
        """
        Cleans and converts specific columns to numerical values, filling NaNs.
        """
        df.fillna(0, inplace=True)
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        return df

    def prepare_features(self, df):
        """
        Prepares feature (X) and target (y) variables.
        """
        feature_columns = [
            'AcresBurned', 'Fatalities', 'Injuries', 'StructuresDamaged', 
            'StructuresDestroyed', 'Duration', 'AirTankers', 'Helicopters',
            'Engines', 'CrewsInvolved', 'Dozers', 'WaterTenders', 'MajorIncident'
        ]
        X = df[feature_columns]
        y = X['AcresBurned']  # Using 'AcresBurned' as a proxy for severity
        return X, y

    def optimize_severity_model(self, X, y, model, param_distributions):
        """
        Performs RandomizedSearchCV to optimize a custom severity estimation model.
        """
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=1000,
            cv=5,
            random_state=42,
            n_jobs=-1,
            scoring='neg_mean_squared_error'
        )
        random_search.fit(X, y)
        self.best_weights = random_search.best_params_
        self.best_estimator = random_search.best_estimator_

    def calculate_severity_scores(self, df, X):
        """
        Calculates and scales severity scores based on optimized model weights.
        """
        df['OptimizedSeverityScore'] = self.best_estimator.predict(X)
        min_score = df['OptimizedSeverityScore'].min()
        df['OptimizedSeverityScore'] += abs(min_score) + 1e-5
        max_score = df['OptimizedSeverityScore'].max()
        df['ScaledSeverityScore'] = 100 * (df['OptimizedSeverityScore'] - min_score) / (max_score - min_score)
        return df

    def expand_fire_rows(self, row):
        """
        Expands rows for fires lasting multiple days into daily entries.
        """
        if row['Duration'] > 0:
            expanded_dates = pd.date_range(row['Started'], periods=row['Duration'])
            expanded_rows = pd.DataFrame({
                'AcresBurned': [round(row['AcresBurned'] / (len(expanded_dates) - i), 3) for i in range(len(expanded_dates))],
                'ArchiveYear': row['ArchiveYear'],
                'Counties': row['Counties'],
                'Latitude': row['Latitude'],
                'Location': row['Location'],
                'Longitude': row['Longitude'],
                'Name': row['Name'],
                'Started': expanded_dates,
                'Extinguished': row['Extinguished'],
                'Duration': [row['Duration'] - i for i in range(len(expanded_dates))],
                'OptimizedSeverityScore': [round(row['OptimizedSeverityScore']/(len(expanded_dates) - i), 3) for i in range(len(expanded_dates))]
            })
            return expanded_rows
        else:
            return pd.DataFrame([row])

    def expand_rows(self, df):
        """
        Applies the expand_fire_rows function to the entire DataFrame.
        """
        expanded_df = pd.concat(df.apply(self.expand_fire_rows, axis=1).tolist(), ignore_index=True)
        return expanded_df

    def preprocess(self, df, model, param_distributions):
        """
        Full preprocessing pipeline.
        """
        df = self.filter_columns(df)
        df = self.process_datetime_columns(df)
        df = self.clean_and_convert(df)
        X, y = self.prepare_features(df)
        self.optimize_severity_model(X, y, model, param_distributions)
        df = self.calculate_severity_scores(df, X)
        expanded_df = self.expand_rows(df)
        expanded_df['OptimizedSeverityScore_Log'] = np.log1p(expanded_df['OptimizedSeverityScore'])
        return expanded_df
