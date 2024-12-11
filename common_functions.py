from common_imports import *

def merge_files(dir_path):
    files = [pd.read_csv(os.path.join(dir_path, filename)) for filename in os.listdir(dir_path) if filename.endswith('.csv')]
    merged_df = pd.concat(files, ignore_index=True)
    return merged_df

def preprocess_df(df):
    # Preprocessing the data
    df['datetime'] = pd.to_datetime(df['datetime'])  # Convert 'date' column to datetime
    df.sort_values(by=['name', 'datetime'], inplace=True)

    scaler = StandardScaler()
    weather_features = ["temp", "dew", "precip", "precipcover",
                        "windspeed","winddir","sealevelpressure",
                        "cloudcover","solarradiation", "elevation"]

    df[weather_features] = scaler.fit_transform(df[weather_features])
    df['FireOccurred'] = df['Started'].notna().astype(int)

    df = df.drop(columns=['windgust', 'Lat', 'Long', 'Duration', 'Started', 'OptimizedSeverityScore', "closest_city"])

    df.fillna(0, inplace=True)
    return df, weather_features

# Severity Calculation

class WildfireSeverityEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, w1=1, w2=1, w3=1, w4 = 1, w5=1, w6=1, w7=1):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6
        self.w7 = w7

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X = X.clip(lower=0)
        
        # Calculate the severity score using the given weights
        score = (
            self.w1 * np.log(X['AcresBurned'] + 1e-5) +  # Added + 1e-5 avoid log(0)
            self.w2 * (X['Fatalities'] + X['Injuries']) +
            self.w3 * (X['StructuresDamaged'] + 2 * X['StructuresDestroyed']) +
            self.w4 * np.log(X['Duration']+ 1e-5) +  # Added + 1e-5 avoid log(0)
            self.w5 * (X['AirTankers'] + X['Helicopters']) +
            self.w6 * (X['Engines'] + X['CrewsInvolved'] + X['Dozers'] + X['WaterTenders']) +
            self.w7 * X['MajorIncident']
        )
        return score
    

# Model Setup
def tt_split(cities, df, sequence_length, weather_features):
    all_X = []
    for city in cities:
        city_df = df[df['name'] == city]
        for i in range(len(city_df) - sequence_length):
            # Extract sequences of weather features
            weather_data = city_df[weather_features].iloc[i:i + sequence_length].values

            # Fire occurrence (binary)
            fire_label = city_df['FireOccurred'].iloc[i + sequence_length]

            # Fire severity (set to 0 if no fire occurred)
            severity_label = (
                city_df['OptimizedSeverityScore_Log'].iloc[i + sequence_length]
                if fire_label == 1 else 0
            )

            associated_date = city_df['datetime'].iloc[i + sequence_length]
            associated_city = city

            all_X.append({
                'city': associated_city,
                'datetime': associated_date,
                'weather_data': weather_data,
                'y_fire': fire_label,
                'y_severity': severity_label
            })
            
    # Convert unified data to DataFrame and numpy arrays
    all_X = pd.DataFrame(all_X)

    return all_X 

def grouped_tt_split(counties, df, sequence_length, weather_features):
    """
    Combine county grouping with centralized X DataFrame.
    Returns grouped data by county and a unified DataFrame for X.
    """
    all_X = []

    for county in counties:
        county_df = df[df['county'] == county]

        for i in range(len(county_df) - sequence_length):
            # Extract weather data sequence
            weather_data = county_df[weather_features].iloc[i:i + sequence_length].values

            # Fire occurrence (binary)
            fire_label = county_df['FireOccurred'].iloc[i + sequence_length]

            # Fire severity (set to 0 if no fire occurred)
            severity_label = (
                county_df['OptimizedSeverityScore_Log'].iloc[i + sequence_length]
                if fire_label == 1 else 0
            )

            associated_date = county_df['datetime'].iloc[i + sequence_length]
            associated_county = county

            # Append to unified lists
            all_X.append({
                'county': associated_county,
                'datetime': associated_date,
                'weather_data': weather_data,
                'y_fire': fire_label,
                'y_severity': severity_label
            })

    # Convert unified data to DataFrame and numpy arrays
    all_X = pd.DataFrame(all_X)

    return all_X 


def train_val_test(train_df, test_df, municipality, sequence_length, weather_features, grouped = False):
    # Validation and Test splits based on time
    validation = test_df[test_df['datetime'] < '2019-01-01']
    test = test_df[test_df['datetime'] >= '2019-01-01']
    
    if grouped == False:
        X_train = tt_split(municipality, train_df, sequence_length, weather_features)
        X_val = tt_split(municipality, validation, sequence_length, weather_features)
        X_test = tt_split(municipality, test, sequence_length, weather_features)
    else:
        X_train = grouped_tt_split(municipality, train_df, sequence_length, weather_features)
        X_val = grouped_tt_split(municipality, validation, sequence_length, weather_features)
        X_test = grouped_tt_split(municipality, test, sequence_length, weather_features)

    return (
        X_train,
        X_val,
        X_test
    )


def setup_x_and_y(train, val):
    train_weather_features = np.stack(train['weather_data'].values)
    val_weather_features = np.stack(val['weather_data'].values)

    y_train_fire = np.stack(train['y_fire'].values)
    y_train_severity = np.stack(train['y_severity'].values)
    y_val_fire = np.stack(val['y_fire'].values)
    y_val_severity = np.stack(val['y_severity'].values)

    return train_weather_features, val_weather_features, y_train_fire, y_train_severity, y_val_fire, y_val_severity


# Imbalance adjustement 
def expand_fire_rows(row):
    if row['Duration'] > 0:
        # Create a range of dates for the fire's duration
        expanded_dates = pd.date_range(row['Started'], periods=row['Duration'])
         # Create a DataFrame for the expanded rows
        expanded_rows = pd.DataFrame({
            'AcresBurned': [round(row['AcresBurned'] / (len(expanded_dates) - i), 3) for i in range(len(expanded_dates))],  # Distribute acres burned equally
            'ArchiveYear': row['ArchiveYear'],
            'Counties': row['Counties'],
            'Latitude': row['Latitude'],
            'Location': row['Location'],
            'Longitude': row['Longitude'],
            'Name': row['Name'],
            'Started': expanded_dates,  # Assign each date in the range
            'Extinguished': row['Extinguished'],
            'Duration': [row['Duration'] - i for i in range(len(expanded_dates))],  # Remaining duration
            'OptimizedSeverityScore': [round(row['OptimizedSeverityScore']/(len(expanded_dates) - i), 3) for i in range(len(expanded_dates))],  # Distribute severity score
        })
        
        return expanded_rows
    else:
        return pd.DataFrame([row])  # If single-day fire or no fire, keep the row as is
    

def count_check(df, column='FireOccurred', return_df=False):
    """
    Count the occurrences of unique values in the specified column of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to count values for. Default is 'FireOccurred'.
        return_df (bool): Whether to return the resulting DataFrame. Default is False.

    Returns:
        pd.DataFrame (optional): A DataFrame with unique values and their counts.
    """
    # Check if the column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    # Count occurrences
    value_counts = df.groupby([column]).size().reset_index(name='Count')
    
    # Print or return the result
    if return_df:
        return value_counts
    else:
        print(value_counts)

def infrequent_over_sample(df, cities):
    """
    Balance the dataset by applying infrequent oversampling or undersampling per city.
    
    Args:
        df (pd.DataFrame): Input dataset containing the data.
        cities (list): List of unique city names.
    
    Returns:
        pd.DataFrame: Balanced dataset.
    """
    balanced_df = pd.DataFrame()

    for city in cities:
        city_df = df[df['name'] == city]  # Filter data for each city

        occurred = city_df[city_df['FireOccurred'] == 1]
        non_occurred = city_df[city_df['FireOccurred'] == 0]

        occurred.sort_values(by='datetime')
        non_occurred.sort_values(by='datetime')

        if len(non_occurred) == 0 or len(occurred) == 0:
            adj_df = city_df

        elif len(non_occurred) > len(occurred):
            ratio = len(non_occurred) / len(occurred)

            # Only undersample if majority class is at least twice the size of minority
            if ratio >= 2:
                step = int(np.floor(ratio))  # Compute the step size
                adj_non_occurred = non_occurred.iloc[::step]
            else:
                adj_non_occurred = non_occurred

            adj_df = pd.concat([occurred, adj_non_occurred], ignore_index=True)

        elif len(occurred) > len(non_occurred):
            ratio = len(occurred) / len(non_occurred)

            # Only oversample if minority class is at least twice smaller
            if ratio >= 2:
                step = int(np.floor(ratio))  # Compute the step size
                adj_occurred = occurred.iloc[::step]
            else:
                adj_occurred = occurred

            adj_df = pd.concat([non_occurred, adj_occurred], ignore_index=True)

        else:
            # If both classes are equal, keep the data as-is
            adj_df = city_df

        # Add balanced data for this city to the final dataset
        balanced_df = pd.concat([balanced_df, adj_df], ignore_index=True)

    return balanced_df

def sliding_over_sample(df, cities):
    """
    Balance the dataset by applying sliding window oversampling or undersampling per city.

    Args:
        df (pd.DataFrame): Input dataset containing the data.
        cities (list): List of unique city names.

    Returns:
        pd.DataFrame: Balanced dataset.
    """
    balanced_df = pd.DataFrame()

    for city in cities:
        city_df = df[df['name'] == city]  # Filter data for each city

        occurred = city_df[city_df['FireOccurred'] == 1]
        non_occurred = city_df[city_df['FireOccurred'] == 0]

        occurred.sort_values(by='datetime')
        non_occurred.sort_values(by='datetime')

        # Skip processing if one of the classes is empty
        if len(non_occurred) == 0 or len(occurred) == 0:
            balanced_df = pd.concat([balanced_df, city_df], ignore_index=True)
            continue

        if len(non_occurred) > len(occurred):
            ratio = len(non_occurred) / len(occurred)
            step = int(np.floor(ratio)) if ratio >= 2 else 1
            aggregated_non_occurred = []

            for i in range(0, len(non_occurred), step):
                window = non_occurred.iloc[i:i + step]
                numeric_agg = window.mean(numeric_only=True)
                name_agg = window['name'].iloc[0]
                datetime_agg = window['datetime'].iloc[0]
                aggregated_non_occurred.append(
                    pd.concat([numeric_agg, pd.Series({'name': name_agg, 'datetime': datetime_agg})])
                )

            adj_non_occurred = pd.DataFrame(aggregated_non_occurred)
            adj_df = pd.concat([occurred, adj_non_occurred], ignore_index=True)

        elif len(occurred) > len(non_occurred):
            ratio = len(occurred) / len(non_occurred)
            step = int(np.floor(ratio)) if ratio >= 2 else 1
            aggregated_occurred = []

            for i in range(0, len(occurred), step):
                window = occurred.iloc[i:i + step]
                numeric_agg = window.mean(numeric_only=True)
                name_agg = window['name'].iloc[0]
                datetime_agg = window['datetime'].iloc[0]
                aggregated_occurred.append(
                    pd.concat([numeric_agg, pd.Series({'name': name_agg, 'datetime': datetime_agg})])
                )

            adj_occurred = pd.DataFrame(aggregated_occurred)
            adj_df = pd.concat([non_occurred, adj_occurred], ignore_index=True)

        else:
            # If classes are balanced, retain as-is
            adj_df = city_df

        # Add adjusted data for the city to the final balanced dataset
        balanced_df = pd.concat([balanced_df, adj_df], ignore_index=True)

    balanced_df['FireOccurred'] = balanced_df['FireOccurred'].astype(int)
    return balanced_df


# Customized performance metricts
def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.where(y_true == 1, 2.0, 1.0)  # Weight minority class higher
    return tf.reduce_mean(weights * tf.keras.losses.binary_crossentropy(y_true, y_pred))

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    loss = -alpha * (1 - pt) ** gamma * tf.math.log(pt + tf.keras.backend.epsilon())
    return tf.reduce_mean(loss)

def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(tf.round(y_pred), 'float32')
    precision = tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_pred) + tf.keras.backend.epsilon())
    recall = tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_true) + tf.keras.backend.epsilon())
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))


def model_performance(X_train, y_train_fire, y_train_severity, X_val, y_val_fire, y_val_severity, model, patience=10, threshold = 0.8):
    """
    Train the model and return history and predictions for training and validation sets.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

    history = model.fit(
        x=X_train, 
        y={'fire_output': y_train_fire, 'severity_output': y_train_severity},
        validation_data=(
            X_val, 
            {'fire_output': y_val_fire, 'severity_output': y_val_severity}
        ),
        epochs=10,
        batch_size=64,
        callbacks=[early_stopping, model_checkpoint]
    )

    # Predict probabilities for fire_output
    y_train_pred_proba_fire = model.predict(X_train, verbose=0)['fire_output']
    y_val_pred_proba_fire = model.predict(X_val, verbose=0)['fire_output']

    # Apply threshold to predictions for train and validation sets
    y_train_pred_proba_fire = (y_train_pred_proba_fire > threshold).astype(int)

    y_train_pred_proba_severity = model.predict(X_train, verbose=0)['severity_output']
    y_val_pred_proba_severity = model.predict(X_val, verbose=0)['severity_output']
    
    return history, y_train_pred_proba_fire, y_val_pred_proba_fire, y_train_pred_proba_severity, y_val_pred_proba_severity

# Performance modeling 
def plot_performance(history, y_train_true, y_train_pred_proba, y_val_true, y_val_pred_proba):
    """
    Plots the performance metrics of the model training and validation,
    including loss, accuracy, and ROC curves.
    
    Parameters:
    - history: Training history object from the model
    - y_train_true: True labels for the training data
    - y_train_pred_proba: Predicted probabilities for the training data
    - y_val_true: True labels for the validation data
    - y_val_pred_proba: Predicted probabilities for the validation data
    """
    # Extract the metrics from the history object
    history_dict = history.history

    # Compute ROC curves and AUC for training and validation
    fpr_train, tpr_train, _ = roc_curve(y_train_true, y_train_pred_proba)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    fpr_val, tpr_val, _ = roc_curve(y_val_true, y_val_pred_proba)
    roc_auc_val = auc(fpr_val, tpr_val)

    # Plotting the metrics
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    # Loss Plot
    ax[0, 0].plot(history_dict['loss'], label='Total Training Loss', color='blue')
    ax[0, 0].plot(history_dict['val_loss'], label='Total Validation Loss', color='orange')
    ax[0, 0].set_title('Training and Validation Loss Over Epochs')
    ax[0, 0].set_xlabel('Epochs')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].legend()
    ax[0, 0].grid(True)
    
    # Accuracy Plot
    ax[0, 1].plot(history_dict['fire_output_accuracy'], label='Training Accuracy', color='green')
    ax[0, 1].plot(history_dict['val_fire_output_accuracy'], label='Validation Accuracy', color='red')
    ax[0, 1].set_title('Training and Validation Accuracy Over Epochs')
    ax[0, 1].set_xlabel('Epochs')
    ax[0, 1].set_ylabel('Accuracy')
    ax[0, 1].legend()
    ax[0, 1].grid(True)

    # ROC Curve Plot
    ax[1, 0].plot(fpr_train, tpr_train, label=f'Training ROC (AUC = {roc_auc_train:.2f})', color='blue')
    ax[1, 0].plot(fpr_val, tpr_val, label=f'Validation ROC (AUC = {roc_auc_val:.2f})', color='orange')
    ax[1, 0].plot([0, 1], [0, 1], 'k--', label='Chance Level')
    ax[1, 0].set_title('ROC Curve')
    ax[1, 0].set_xlabel('False Positive Rate')
    ax[1, 0].set_ylabel('True Positive Rate')
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    ax[1, 1].plot(history_dict['fire_output_f1_score'], label='Training F-1 Score', color='green')
    ax[1, 1].plot(history_dict['val_fire_output_f1_score'], label='Validation F-1 Score', color='red')
    ax[1, 1].set_title('Training and Validation F-1 Over Epochs')
    ax[1, 1].set_xlabel('Epochs')
    ax[1, 1].set_ylabel('F1-score')
    ax[1, 1].legend()
    ax[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

# plot prediction preformance
def plot_result_df(df):
    # ROC curve and AUC computation
    fpr, tpr, thresholds = roc_curve(df['FireTrue'], df['FirePrediction'])
    roc_auc = auc(fpr, tpr)

    # Create a figure with 2x2 grid layout
    plt.figure(figsize=(14, 10))

    # Plot Fire Prediction vs. True Fire in histogram (1st subplot)
    plt.subplot(2, 2, 1)
    plt.hist(df['FireTrue'], bins=30, alpha=0.7, label='True Fire', color='blue')
    plt.hist(df['FirePrediction'], bins=30, alpha=0.7, label='Predicted Fire Probability', color='orange')
    plt.title('Fire Prediction vs True Fire Distribution')
    plt.xlabel('Fire Occurrence (True) / Fire Prediction (Probability)')
    plt.ylabel('Frequency')
    plt.yscale('log')  # Log scale on y-axis
    plt.legend()
    plt.grid(True)

    # Plot Fire Severity Prediction vs True Severity in histogram (2nd subplot)
    plt.subplot(2, 2, 2)
    plt.hist(df['SeverityTrue'], bins=30, alpha=0.7, label='True Severity', color='blue')
    plt.hist(df['SeverityPrediction'], bins=30, alpha=0.7, label='Predicted Severity', color='orange')
    plt.title('Severity Prediction vs True Severity Distribution')
    plt.xlabel('Severity')
    plt.ylabel('Frequency')
    plt.yscale('log')  # Log scale on y-axis
    plt.legend()
    plt.grid(True)

    # Plot ROC Curve (3rd subplot)
    plt.subplot(2, 2, 3)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

def plot_best_model_performance(histories, y_train_true, y_train_pred_proba, y_val_true, y_val_pred_proba, counties, threshold=0.8):
    """
    Plots the performance metrics of the model training and validation,
    including loss, accuracy, and ROC curves for each county.
    
    Parameters:
    - histories: Dictionary of training history metrics for each county.
    - counties: List of county names to plot performance for.
    - threshold: Threshold for binary classification.
    """

    # Use the dictionary for plotting
    for idx, county in enumerate(counties):
        if county not in histories:
            print(f"No history data for county: {county}")
            continue

        metrics = histories[county]
        print(f"Processing data for county: {county}")
   
        # Compute ROC curves and AUC for training and validation
        fpr_train, tpr_train, _ = roc_curve(y_train_true[idx], y_train_pred_proba[idx])
        roc_auc_train = auc(fpr_train, tpr_train)
        
        fpr_val, tpr_val, _ = roc_curve(y_val_true[idx], y_val_pred_proba[idx])
        roc_auc_val = auc(fpr_val, tpr_val)

        # Plotting the metrics for each county
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))

        # Loss Plot
        ax[0, 0].plot(metrics['loss'], label='Training Loss', color='blue')
        ax[0, 0].plot(metrics['val_loss'], label='Validation Loss', color='orange')
        ax[0, 0].set_title(f'{county} - Training and Validation Loss Over Epochs')
        ax[0, 0].set_xlabel('Epochs')
        ax[0, 0].set_ylabel('Loss')
        ax[0, 0].legend()
        ax[0, 0].grid(True)

        # Accuracy Plot
        ax[0, 1].plot(metrics['fire_output_accuracy'], label='Training Accuracy', color='green')
        ax[0, 1].plot(metrics['val_fire_output_accuracy'], label='Validation Accuracy', color='red')
        ax[0, 1].set_title(f'{county} - Training and Validation Accuracy Over Epochs')
        ax[0, 1].set_xlabel('Epochs')
        ax[0, 1].set_ylabel('Accuracy')
        ax[0, 1].legend()
        ax[0, 1].grid(True)

        # ROC Curve Plot (placeholders; real values needed)
        if fpr_train.size > 0 and tpr_train.size > 0:
            ax[1, 0].plot(fpr_train, tpr_train, label=f'Training ROC (AUC = {roc_auc_train:.2f})', color='blue')
            ax[1, 0].plot(fpr_val, tpr_val, label=f'Validation ROC (AUC = {roc_auc_val:.2f})', color='orange')
            ax[1, 0].plot([0, 1], [0, 1], 'k--', label='Chance Level')
            ax[1, 0].set_title(f'{county} - ROC Curve')
            ax[1, 0].set_xlabel('False Positive Rate')
            ax[1, 0].set_ylabel('True Positive Rate')
            ax[1, 0].legend()
            ax[1, 0].grid(True)
        else:
            ax[1, 0].axis('off')  # Hide plot if data is unavailable

        # F1 Score Plot
        if 'fire_output_f1_score' in metrics:
            ax[1, 1].plot(metrics['fire_output_f1_score'], label='Training F1 Score', color='green')
            ax[1, 1].plot(metrics['val_fire_output_f1_score'], label='Validation F1 Score', color='red')
            ax[1, 1].set_title(f'{county} - Training and Validation F1 Over Epochs')
            ax[1, 1].set_xlabel('Epochs')
            ax[1, 1].set_ylabel('F1 Score')
            ax[1, 1].legend()
            ax[1, 1].grid(True)
        else:
            ax[1, 1].axis('off')  # Hide plot if F1 data is unavailable

        plt.tight_layout()
        plt.show()


# model selection 
# Best Performance model by val_loss
def county_model_performance_best_params(X_train, X_val, best_results, patience=10, threshold=0.8, sequence_length = 15):
    """
    Train models for each county using the best parameters and return history and predictions.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    all_histories = {}
    y_train_true_fire = []
    y_train_pred_fire = []
    y_val_true_fire = []
    y_val_pred_fire = []
    y_train_true_severity = []
    y_train_pred_severity = []
    y_val_true_severity = []
    y_val_pred_severity = []

    for county, result in best_results.items():
        print(f"Training model for county: {county}")

        best_params = result['params']

        # Filter data for the current county
        county_train_data = X_train[X_train["county"] == county]
        county_val_data = X_val[X_val["county"] == county]

        if county_train_data.empty or county_val_data.empty:
            print(f"No data available for county: {county}")
            continue

        # Setup features and targets
        train_weather_features, val_weather_features, y_train_fire, y_train_severity, y_val_fire, y_val_severity = setup_x_and_y(county_train_data, county_val_data)

        # Create model with best parameters
        model = create_lstm_model(
            sequence_length=sequence_length, 
            weather_features=train_weather_features.shape[2],
            **best_params
        )

        # Correct the filepath formatting for ModelCheckpoint
        checkpoint_filepath = f'best_model_{county}.keras'
        model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_loss')

        # Train the model
        county_history = model.fit(
            x=train_weather_features,
            y={'fire_output': y_train_fire, 'severity_output': y_train_severity},
            validation_data=(
                val_weather_features,
                {'fire_output': y_val_fire, 'severity_output': y_val_severity}
            ),
            epochs=50,
            batch_size=64,
            callbacks=[early_stopping, model_checkpoint],
            verbose=0
        )

        # Store training history
        all_histories[county] = county_history.history
        
        # True values
        y_train_true_fire.append(y_train_fire)
        y_val_true_fire.append(y_val_fire)
        y_train_true_severity.append(y_train_severity)
        y_val_true_severity.append(y_val_severity)

        # Predict probabilities
        y_train_pred_fire_proba = model.predict(train_weather_features, verbose=0)['fire_output']
        y_val_pred_fire_proba = model.predict(val_weather_features, verbose=0)['fire_output']

        # Apply threshold for fire output
        y_train_pred_fire.append((y_train_pred_fire_proba > threshold).astype(int))
        y_val_pred_fire.append((y_val_pred_fire_proba > threshold).astype(int))

        # Predict severity output
        y_train_pred_severity.append(model.predict(train_weather_features, verbose=0)['severity_output'])
        y_val_pred_severity.append(model.predict(val_weather_features, verbose=0)['severity_output'])

    return all_histories, y_train_true_fire, y_train_pred_fire, y_val_true_fire, y_val_pred_fire, y_train_true_severity, y_train_pred_severity, y_val_true_severity, y_val_pred_severity


def model_performance_best_params(X_train, X_val, best_results, patience=10, threshold=0.8, sequence_length = 15):
    """
    Train models for each county using the best parameters and return history and predictions.
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    all_histories = []
    y_train_true_fire = []
    y_train_pred_fire = []
    y_val_true_fire = []
    y_val_pred_fire = []
    y_train_true_severity = []
    y_train_pred_severity = []
    y_val_true_severity = []
    y_val_pred_severity = []

    best_params = best_results['params']

    # Setup features and targets
    train_weather_features, val_weather_features, y_train_fire, y_train_severity, y_val_fire, y_val_severity = setup_x_and_y(X_train, X_val)

    # Create model with best parameters
    model = create_lstm_model(
        sequence_length=sequence_length, 
        weather_features=train_weather_features.shape[2],
        **best_params
    )

    # Correct the filepath formatting for ModelCheckpoint
    checkpoint_filepath = f'best_model_undersample.keras'
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_loss')

    # Train the model
    model_history = model.fit(
        x=train_weather_features,
        y={'fire_output': y_train_fire, 'severity_output': y_train_severity},
        validation_data=(
            val_weather_features,
            {'fire_output': y_val_fire, 'severity_output': y_val_severity}
        ),
        epochs=50,
        batch_size=64,
        callbacks=[early_stopping, model_checkpoint],
        verbose=0
    )

    # Store training history
    all_histories.append(model_history)
    
    # True values
    y_train_true_fire.append(y_train_fire)
    y_val_true_fire.append(y_val_fire)
    y_train_true_severity.append(y_train_severity)
    y_val_true_severity.append(y_val_severity)

    # Predict probabilities
    y_train_pred_fire_proba = model.predict(train_weather_features, verbose=0)['fire_output']
    y_val_pred_fire_proba = model.predict(val_weather_features, verbose=0)['fire_output']

    # Apply threshold for fire output
    y_train_pred_fire.append((y_train_pred_fire_proba > threshold).astype(int))
    y_val_pred_fire.append((y_val_pred_fire_proba > threshold).astype(int))

    # Predict severity output
    y_train_pred_severity.append(model.predict(train_weather_features, verbose=0)['severity_output'])
    y_val_pred_severity.append(model.predict(val_weather_features, verbose=0)['severity_output'])

    return all_histories, y_train_true_fire, y_train_pred_fire, y_val_true_fire, y_val_pred_fire, y_train_true_severity, y_train_pred_severity, y_val_true_severity, y_val_pred_severity


# Randomized Grid Search 
def randomized_cv_search(X_train, X_val, counties, param_grid, n_iter=10, cv_folds=5):
    results = []

    # Setup features and labels
    train_weather_features, val_weather_features, y_train_fire, y_train_severity, y_val_fire, y_val_severity = setup_x_and_y(X_train, X_val)

    # Define TimeSeriesSplit iterator (don't shuffle data)
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_weather_features)):
        # Split into training and validation for this fold
        fold_train_features = train_weather_features[train_idx]
        fold_val_features = train_weather_features[val_idx]
        fold_y_train_fire = y_train_fire[train_idx]
        fold_y_val_fire = y_train_fire[val_idx]
        fold_y_train_severity = y_train_severity[train_idx]
        fold_y_val_severity = y_train_severity[val_idx]

        for _ in range(n_iter):
            # Randomly sample hyperparameters
            params = {k: random.choice(v) for k, v in param_grid.items()}

            # Create and train the model
            model = create_lstm_model(
                sequence_length=fold_train_features.shape[1],
                weather_features=fold_train_features.shape[2],
                **params
            )

            try:
                model.fit(
                    x=fold_train_features,
                    y={'fire_output': fold_y_train_fire, 'severity_output': fold_y_train_severity},
                    validation_data=(fold_val_features, {'fire_output': fold_y_val_fire, 'severity_output': fold_y_val_severity}),
                    epochs=10,
                    batch_size=64,
                    verbose=2
                )

                # Evaluate the model
                eval_metrics = model.evaluate(
                    fold_val_features, 
                    {'fire_output': fold_y_val_fire, 'severity_output': fold_y_val_severity},
                    verbose=0
                )

                # Dynamically extract all metrics based on the compiled model's outputs
                metric_names = model.metrics_names  # Retrieve the names of the metrics
                metrics_dict = {name: value for name, value in zip(metric_names, eval_metrics)}

                # Add to results, organizing metrics by task
                results.append({
                    'fold': fold,
                    'params': params,
                    'metrics': metrics_dict
                })

            except Exception as e:
                print(f"Error training or evaluating fold {fold}: {e}")
                continue

    return results

def randomized_cv_search_group(X_train, X_val, counties, param_grid, n_iter=10, cv_folds=5):
    results = []

    for county in counties:
        county_train_data = X_train[X_train["county"] == county]
        county_val_data = X_val[X_val["county"] == county]

        if county_train_data.empty or county_val_data.empty:
            print(f"No data available for county: {county}")
            continue  # Skip this county if no data is available

        try:
            train_weather_features, val_weather_features, y_train_fire, y_train_severity, y_val_fire, y_val_severity =  setup_x_and_y(county_train_data, county_val_data)
            
        except ValueError as e:
            print(f"Error processing data for county {county}: {e}")
            continue

        tscv = TimeSeriesSplit(n_splits=cv_folds)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(train_weather_features)):
            # Split into training and validation for this fold
            fold_train_features = train_weather_features[train_idx]
            fold_val_features = train_weather_features[val_idx]
            fold_y_train_fire = y_train_fire[train_idx]
            fold_y_val_fire = y_train_fire[val_idx]
            fold_y_train_severity = y_train_severity[train_idx]
            fold_y_val_severity = y_train_severity[val_idx]

            for _ in range(n_iter):
                # Randomly sample hyperparameters
                params = {k: random.choice(v) for k, v in param_grid.items()}

                # Create and train the model
                model = create_lstm_model(
                sequence_length=fold_train_features.shape[1],
                weather_features=fold_train_features.shape[2],
                **params
            )

                try:
                    model.fit(
                        x=fold_train_features,
                        y={'fire_output': fold_y_train_fire, 'severity_output': fold_y_train_severity},
                        validation_data=(fold_val_features, {'fire_output': fold_y_val_fire, 'severity_output': fold_y_val_severity}),
                        epochs=10,
                        batch_size=64,
                        verbose=2
                    )

                    # Evaluate the model
                    eval_metrics = model.evaluate(
                        fold_val_features, 
                        {'fire_output': fold_y_val_fire, 'severity_output': fold_y_val_severity},
                        verbose=0
                    )

                    # Dynamically extract all metrics based on the compiled model's outputs
                    metric_names = model.metrics_names  # Retrieve the names of the metrics
                    metrics_dict = {name: value for name, value in zip(metric_names, eval_metrics)}

                    # Add to results, organizing metrics by task
                    results.append({
                        'county': county,
                        'fold': fold,
                        'params': params,
                        'metrics': metrics_dict
                    })
                except Exception as e:
                    print(f"Error training or evaluating for county {county}: {e}")
                    continue

    return results

# Models 
# Function to create an LSTM model
def create_lstm_model(sequence_length, weather_features, lstm_units, dropout_rates, batch_norm, reg_l2, learning_rate):
    inputs = Input(shape=(sequence_length, weather_features), name="input_layer")
    x = inputs

    # LSTM layers
    for i, units in enumerate(lstm_units):
        x = LSTM(units, return_sequences=(i < len(lstm_units) - 1), 
                 kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = Dropout(dropout_rates)(x)
    
    # Separate outputs
    fire_output = Dense(1, activation='sigmoid', name='fire_output')(x)
    severity_output = Dense(1, activation='linear', name='severity_output')(x)

    # Create model
    model = Model(inputs=inputs, outputs={'fire_output': fire_output, 'severity_output': severity_output})

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={'fire_output': 'binary_crossentropy', 'severity_output': 'mse'},
        metrics={'fire_output': [f1_score, 'accuracy'], 'severity_output': ['mae']},
        loss_weights={'fire_output': 0.7, 'severity_output': 0.3}
    )
    return model

def LSTM_model(sequence_length, weather_features, lstm_units=[64, 128], dropout_rates=None, batch_norm=False, reg_l2=None):
    """
    Builds an LSTM model for wildfire prediction.

    Parameters:
    - sequence_length: Length of input sequences
    - num_features: Number of features in input data
    - lstm_units: List of integers specifying LSTM layer sizes
    - dropout_rates: List of dropout rates (optional)
    - batch_norm: Boolean flag for batch normalization
    - reg_l2: L2 regularization factor (optional)

    Returns:
    - A Keras model object
    """
    if dropout_rates is None:
        dropout_rates = [0.0] * len(lstm_units)  # Default to no dropout if not specified

    input_layer = Input(shape=(sequence_length, len(weather_features)), name='weather_input')

    x = input_layer
    for i, units in enumerate(lstm_units):
        kernel_regularizer = l2(reg_l2) if reg_l2 else None
        x = LSTM(
            units=units, 
            return_sequences=(i < len(lstm_units) - 1), 
            kernel_regularizer=kernel_regularizer
        )(x)
        
        if batch_norm:
            x = BatchNormalization()(x)
        
        if dropout_rates:
            x = Dropout(rate=dropout_rates[i])(x)

    # Output for fire occurrence (binary classification)
    fire_output = Dense(1, activation='sigmoid', name='fire_output')(x)
    # Output for fire severity (regression)
    severity_output = Dense(1, activation='relu', name='severity_output')(x)

    model = Model(inputs=input_layer, outputs={'fire_output': fire_output, 'severity_output': severity_output})

    return model


# Predictions 
def predict_wildfire(test, fire, severity, model, threshold = 0.13):
    test_array = np.stack(test['weather_data'].values)

    predictions = model.predict(test_array)

    fire_predictions = predictions['fire_output'].flatten()  # Flatten to 1D array
    severity_predictions = predictions['severity_output'].flatten()  # Flatten to 1D array

    # Apply threshold to predictions for train and validation sets
    fire_predictions = (fire_predictions > threshold).astype(int)

    fire = fire.flatten()
    severity = severity.flatten()
    
    fire = fire.astype(int)  # Explicitly convert to integers
    
    # Associate predictions with datetime and city from X_test
    results_df = test.copy()  # Make a copy of test DataFrame for results

    # Add predictions to the DataFrame
    results_df['FirePrediction'] = fire_predictions
    results_df['FireTrue'] = fire
    results_df['SeverityPrediction'] = severity_predictions
    results_df['SeverityTrue'] = severity

    # Optionally, reset index and select relevant columns for interpretability
    results_df = results_df[['datetime', 'city', 
                             'FirePrediction', 'FireTrue',
                             'SeverityPrediction', 'SeverityTrue']]

    # Preview the results
    print(results_df.head())
    print(confusion_matrix(results_df["FireTrue"], results_df["FirePrediction"]))
    
    return results_df