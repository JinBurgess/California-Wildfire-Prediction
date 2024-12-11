from main import *

class CountyModelTrainer:
    def __init__(self, patience=10, threshold=0.8, sequence_length=15):
        """
        Initialize the CountyModelTrainer with key parameters.
        
        Parameters:
        - patience: Early stopping patience for model training.
        - threshold: Threshold for fire prediction binary classification.
        - sequence_length: Length of the input time sequences.
        """
        self.patience = patience
        self.threshold = threshold
        self.sequence_length = sequence_length

    def train_county_models(self, X_train, X_val, json_file, type = "norm"):
        """
        Train LSTM models for each county using the best parameters.

        Parameters:
        - X_train: Training dataset.
        - X_val: Validation dataset.
        - best_results: Dictionary containing the best parameters for each county.

        Returns:
        - Histories and predictions for each county.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        all_histories = {}
        y_train_true_fire, y_train_pred_fire = [], []
        y_val_true_fire, y_val_pred_fire = [], []
        y_train_true_severity, y_train_pred_severity = [], []
        y_val_true_severity, y_val_pred_severity = [], []

        with open(json_file, 'r') as f:
            best_results = json.load(f)

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
            train_weather_features, y_train_fire, y_train_severity = data_setup.setup_x_and_y(county_train_data)
            val_weather_features, y_val_fire, y_val_severity = data_setup.setup_x_and_y(county_val_data)

            # Create model
            model = lstm_blocks.create_lstm_model(
                sequence_length=self.sequence_length,
                weather_features=train_weather_features.shape[2],
                **best_params
            )

            # Checkpoint for saving the best model
            checkpoint_filepath = f'model_performance/log/keras/best_model_{type}_{county}.keras'
            model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_loss')

            # Train the model
            history = model.fit(
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

            all_histories[county] = history.history

           # True values
            y_train_true_fire.append(y_train_fire)
            y_val_true_fire.append(y_val_fire)
            y_train_true_severity.append(y_train_severity)
            y_val_true_severity.append(y_val_severity)

            # Predict probabilities
            y_train_pred_fire_proba = model.predict(train_weather_features, verbose=0)['fire_output']
            y_val_pred_fire_proba = model.predict(val_weather_features, verbose=0)['fire_output']

            # Apply threshold for fire output
            y_train_pred_fire.append((y_train_pred_fire_proba > self.threshold).astype(int))
            y_val_pred_fire.append((y_val_pred_fire_proba > self.threshold).astype(int))

            # Predict severity output
            y_train_pred_severity.append(model.predict(train_weather_features, verbose=0)['severity_output'])
            y_val_pred_severity.append(model.predict(val_weather_features, verbose=0)['severity_output'])

        return (
            all_histories,
            y_train_true_fire, y_train_pred_fire,
            y_val_true_fire, y_val_pred_fire,
            y_train_true_severity, y_train_pred_severity,
            y_val_true_severity, y_val_pred_severity
        )

    def randomized_cv_search(self, X_train, counties, param_grid, n_iter=10, cv_folds=3):
        """
        Perform randomized cross-validation search for hyperparameter tuning.

        Parameters:
        - X_train: Training dataset.
        - X_val: Validation dataset.
        - counties: List of counties to process.
        - param_grid: Dictionary of hyperparameter options.
        - n_iter: Number of iterations for random sampling.
        - cv_folds: Number of cross-validation folds.

        Returns:
        - A list of results containing fold, parameters, and evaluation metrics.
        """
        results = []
        # Define TimeSeriesSplit iterator (don't shuffle data)
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        for county in counties:
            county_train_data = X_train[X_train["county"] == county]

            if county_train_data.empty:
                print(f"No data available for county: {county}")
                continue  # Skip this county if no data is available

            try:
                # Setup features and labels
                train_weather_features, y_train_fire, y_train_severity = data_setup.setup_x_and_y(county_train_data)

            except ValueError as e:
                print(f"Error processing data for county {county}: {e}")
                continue

            print(f"CV for county: {county}")

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
                    model = lstm_blocks.create_lstm_model(
                        sequence_length=sequence_length,
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
                            verbose=0
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
                        print(f"Error training or evaluating for county {county}, fold {fold}: {e}")
                        continue

        return results
