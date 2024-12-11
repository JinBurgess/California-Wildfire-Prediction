from main import *

from lib.lstm_model_trainer import ModelTrainer
from lib.lstm_model_county import CountyModelTrainer

lstm_model = ModelTrainer()
lstm_county_model = CountyModelTrainer()

# Base Approach
#______________________________________________________________________________________________________________________________
# X_train_city, X_val_city, X_test_city = data_setup.train_val_test_split(train_df, test_df, cities, split_by="name")
# json_file = "model_performance/log/json/best_norm_params.json"
# histories, train_true_fire, train_pred_fire, val_true_fire, val_pred_fire, y_train_true_severity, y_train_pred_severity, y_val_true_severity, y_val_pred_severity = lstm_model.train_model(X_train_city, X_val_city, json_file, type = "norm")
# single_history = histories[0]

# plot_perform.plot_performance(
#     single_history,  # Pass the `.history` attribute directly
#     y_train_true=train_true_fire[0],
#     y_train_pred_proba=train_pred_fire[0],
#     y_val_true=val_true_fire[0],
#     y_val_pred_proba=val_pred_fire[0],
#     dir_out = "model_performance/LSTM/norm"
# )

# Minority Weight Approach
#______________________________________________________________________________________________________________________________

# X_train_city, X_val_city, X_test_city = data_setup.train_val_test_split(train_df, test_df, cities, split_by="name")
# json_file_minor_weight = "model_performance/log/json/best_minor_weight_params.json"
# histories, train_true_fire, train_pred_fire, val_true_fire, val_pred_fire, y_train_true_severity, y_train_pred_severity, y_val_true_severity, y_val_pred_severity = lstm_model.train_model(X_train_city, X_val_city, json_file_minor_weight, type = "minor_weight", minor_weight = True)
# single_history = histories[0]

# plot_perform.plot_performance(
#     single_history,  # Pass the `.history` attribute directly
#     y_train_true=train_true_fire[0],
#     y_train_pred_proba=train_pred_fire[0],
#     y_val_true=val_true_fire[0],
#     y_val_pred_proba=val_pred_fire[0],
#     dir_out = "model_performance/LSTM/minor_weight"
# )

# Infrequent Sampling Approach
#______________________________________________________________________________________________________________________________
# balanced_infrequent = balancing.infrequent_over_sample(cities, train_df, by = "name")

# X_train_infreq, X_val_infreq, X_test_infreq = data_setup.train_val_test_split(balanced_infrequent, test_df, cities, split_by="name")
# json_file_infreq = "model_performance/log/json/best_infreq_params.json"
# histories, train_true_fire, train_pred_fire, val_true_fire, val_pred_fire, y_train_true_severity, y_train_pred_severity, y_val_true_severity, y_val_pred_severity = lstm_model.train_model(X_train_infreq, X_val_infreq, json_file_infreq, type = "infreq")
# single_history = histories[0]

# plot_perform.plot_performance(
#     single_history,  # Pass the `.history` attribute directly
#     y_train_true=train_true_fire[0],
#     y_train_pred_proba=train_pred_fire[0],
#     y_val_true=val_true_fire[0],
#     y_val_pred_proba=val_pred_fire[0],
#     dir_out = "model_performance/LSTM/infreq"
# )

# Sliding Window Approach
#______________________________________________________________________________________________________________________________
balanced_sliding = balancing.sliding_over_sample(cities, train_df, by = "name")
X_train_slide, X_val_slide, X_test_slide = data_setup.train_val_test_split(balanced_sliding, test_df, cities, split_by="name")
json_file_slide = "model_performance/log/json/best_slide_params.json"
histories, train_true_fire, train_pred_fire, val_true_fire, val_pred_fire, y_train_true_severity, y_train_pred_severity, y_val_true_severity, y_val_pred_severity = lstm_model.train_model(X_train_slide, X_val_slide, json_file_slide, type = "slide")
single_history = histories[0]

plot_perform.plot_performance(
    single_history,  # Pass the `.history` attribute directly
    y_train_true=train_true_fire[0],
    y_train_pred_proba=train_pred_fire[0],
    y_val_true=val_true_fire[0],
    y_val_pred_proba=val_pred_fire[0],
    dir_out = "model_performance/LSTM/slide"
)

# Regional Grouping Approach
#______________________________________________________________________________________________________________________________
# X_train_county, X_val_county, X_test_county = data_setup.train_val_test_split(train_df, test_df, counties, split_by="county")
# json_file_county = "model_performance/log/json/best_norm_params_county.json"
# histories, train_true_fire, train_pred_fire, val_true_fire, val_pred_fire, y_train_true_severity, y_train_pred_severity, y_val_true_severity, y_val_pred_severity = lstm_county_model.train_county_models(X_train_county, X_val_county, json_file_county, type = "norm")

# plot_perform.plot_best_model_performance(
#     histories, 
#     y_train_true=train_true_fire,
#     y_train_pred_proba=train_pred_fire,
#     y_val_true=val_true_fire,
#     y_val_pred_proba=val_pred_fire,
#     counties=counties,
#     dir_output="model_performance/LSTM_county"
# )