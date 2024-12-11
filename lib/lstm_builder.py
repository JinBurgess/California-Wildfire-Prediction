from main import *
from lib.custom_loss import CustomLoss

custom_metrics = CustomLoss()

class LSTM_Builder:
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

    @staticmethod
    def create_lstm_model(sequence_length, weather_features, lstm_units, dropout_rates, 
                          batch_norm, reg_l2, learning_rate, minor_weight = None):
        """
        Create an LSTM model for time-series predictions.

        Parameters:
        - sequence_length: Length of the input time sequences.
        - weather_features: Number of features per time step.
        - lstm_units: List of LSTM units for each layer.
        - dropout_rates: Dropout rate for regularization.
        - batch_norm: Whether to apply batch normalization.
        - reg_l2: L2 regularization factor.
        - learning_rate: Learning rate for the optimizer.

        Returns:
        - A compiled Keras model.
        """
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

        model = Model(inputs=inputs, outputs={'fire_output': fire_output, 'severity_output': severity_output})

        # Create model
        if minor_weight == None:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss={'fire_output': "binary_crossentropy", 'severity_output': 'mse'},
                metrics={'fire_output': [custom_metrics.f1_score, 'accuracy'], 'severity_output': ['mae']},
                loss_weights={'fire_output': 0.7, 'severity_output': 0.3}
            )
        else:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss={'fire_output':  custom_metrics.dynamic_weighted_binary_crossentropy, 'severity_output': 'mse'},
                metrics={'fire_output': [custom_metrics.f1_score, 'accuracy'], 'severity_output': ['mae']},
                loss_weights={'fire_output': 0.7, 'severity_output': 0.3}
            )
        return model
    
    def return_best_result(self, results, type='norm'):
        # Find the overall best result based on fire_output_loss
        best_results = min(results, key=lambda x: x['metrics']['fire_output_loss'])
        
        # Save the best result to a JSON file
        with open(f'model_performance/log/json/best_{type}_params.json', 'w') as f:
            json.dump(best_results, f, indent=4)
        
        return best_results

    def return_best_result_county(self, results, counties, type='norm'):
        # Dictionary to store the best result for each county
        best_results_loss = {}
        
        # Iterate over each county to find its best result
        for county in counties:
            county_results = [r for r in results if r.get('county') == county]
            if county_results:  # Ensure there are results for the county
                best_results_loss[county] = min(county_results, key=lambda x: x['metrics']['fire_output_loss'])
        
        # Save the best results by county to a JSON file
        with open(f'model_performance/log/json/best_{type}_params_county.json', 'w') as f:
            json.dump(best_results_loss, f, indent=4)
        
        return best_results_loss
