
from common_imports import *

# internals
from lib.wildfire_preprocessing import WildfirePreprocessor
from lib.wildfire_severity_estimator import WildfireSeverityEstimator

# Instantiate the class
wildfire_preprocessor = WildfirePreprocessor()
wildfire_severity_est = WildfireSeverityEstimator()

# Process wildfire data and calculates a severity score based on historic wildfire data
wildfire_df = pd.read_csv("wildfire_data/California_Fire_Incidents.csv")

# Define the parameter grid
param_distributions = {
    'w1': uniform(0, 10),
    'w2': uniform(0, 10),
    'w3': uniform(0, 10),
    'w4': uniform(0, 10),
    'w5': uniform(0, 10),
    'w6': uniform(0, 10),
    'w7': uniform(0, 10)
}

# Preprocess the data
processed_fire_df = wildfire_preprocessor.preprocess(wildfire_df, WildfireSeverityEstimator(), param_distributions)

# Save to CSV
processed_fire_df.to_csv("wildfire_data/wildfire_with_severity.csv", index=False)
