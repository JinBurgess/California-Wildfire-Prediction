from main import *

class WildfirePredictor:
    def __init__(self, threshold=0.5):
        """
        Initializes the WildfirePredictor with the given model and threshold.

        Parameters:
        - threshold: Threshold for predicting fire (default is 0.13).
        """
        self.threshold = threshold

    def predict_func(self, test, json_file, save = None, minor_weight = False):
        """
        Predicts wildfire fire and severity based on the test data and compares them with true values.
        
        Parameters:
        - test: The DataFrame containing test data (with 'weather_data' feature).
        - fire: The true fire labels.
        - severity: The true severity labels.
        
        Returns:
        - results_df: DataFrame with predictions and true labels.
        """

        with open(json_file, 'r') as f:
            best_params = json.load(f)
        best_params = best_params['params']

        # Convert weather data to numpy array for model input
        test_array, fire, severity = data_setup.setup_x_and_y(test)

        # model = lstm_blocks.create_lstm_model(
        #     sequence_length= sequence_length, 
        #     weather_features=test_array.shape[2],
        #     **best_params
        # )
        if minor_weight == False:
            # Create and train the model
            model = lstm_blocks.create_lstm_model(
                sequence_length=sequence_length,
                weather_features=test_array.shape[2],
                **best_params
            )
        else:
            model = lstm_blocks.create_lstm_model(
                sequence_length=sequence_length,
                weather_features=test_array.shape[2],
                **best_params,
                minor_weight = True
            )
        # Make predictions using the model
        predictions = model.predict(test_array)

        # Extract fire and severity predictions
        fire_predictions = predictions['fire_output'].flatten()  # Flatten to 1D array
        severity_predictions = predictions['severity_output'].flatten()  # Flatten to 1D array

        # Apply threshold to fire predictions
        fire_prediction_binary = (fire_predictions > self.threshold).astype(int)

        # Flatten true labels for fire and severity
        fire = fire.flatten().astype(int)
        severity = severity.flatten()

        # Create a DataFrame to store the results
        results_df = test.copy()  # Make a copy of the test DataFrame

        # Add predictions and true labels to the DataFrame
        results_df['FirePredictionBinary'] = fire_prediction_binary
        results_df['FirePrediction'] = fire_predictions
        results_df['FireTrue'] = fire
        results_df['SeverityPrediction'] = severity_predictions
        results_df['SeverityTrue'] = severity

        results_df['SeverityPrediction'] = results_df['SeverityPrediction'] - results_df['SeverityPrediction'].min()

        # Select relevant columns for the results
        results_df = results_df[['datetime', 'name', 
                                 'FirePrediction', 'FirePredictionBinary', 'FireTrue',
                                 'SeverityPrediction', 'SeverityTrue']]

        # Optionally print results and confusion matrix
        print(results_df.head())
        print(confusion_matrix(results_df["FireTrue"], results_df["FirePredictionBinary"]))
        
        if save != None:
            results_df.to_csv(f'{save}.csv', index = False)
        return results_df