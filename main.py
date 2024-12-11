from common_imports import *

# internals
from lib.data_processing import DataProcessor
from lib.data_training_setup import DataSetup
from lib.rebalancing_methods import BalancingMethods
from lib.lstm_builder import LSTM_Builder
from lib.plotting import PerformanceModeling
# from lib.model_prediction import WildfirePredictor

sequence_length = 15  
weather_features = ["temp", "dew", "precip", "precipcover",
                            "windspeed","winddir","sealevelpressure",
                            "cloudcover","solarradiation", "elevation"]

param_grid = {
    'lstm_units': [(64,), (64, 128), (128,), (32, 64), (32,), (32,64, 128), 
                   (64, 128, 512), (32, 64, 64, 128), (32,64,128,256), (64, 64, 128, 256)],
    'dropout_rates': [0.2, 0.3, 0.4],
    'batch_norm': [True, False],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'reg_l2': [0.01, 0.001, 0.0001]
}

# Initialize classes
processor = DataProcessor()
data_setup = DataSetup(sequence_length=sequence_length, weather_features=weather_features)
balancing = BalancingMethods()
lstm_blocks = LSTM_Builder()
plot_perform = PerformanceModeling()
# fire_prediction = WildfirePredictor()

df = pd.read_csv("lookup.csv")
df = processor.preprocess_df(df, weather_features)

# split data into train and test
train_df = df[(df['datetime'] < '2018-01-01')]
test_df = df[(df['datetime'] >= '2018-01-01')]

# Define locations (cities or counties)
cities = train_df['name'].unique()
counties = train_df['county'].unique()

