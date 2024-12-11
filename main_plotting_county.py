from main import *
from main_lstm import *

from lib.lstm_model_county import CountyModelTrainer
lstm_model = CountyModelTrainer()

#______________________________________________________________________________________________________________________________
X_train_county, X_val_county, X_test_county = data_setup.train_val_test_split(train_df, test_df, counties, split_by="county")
json_file = "model_performance/log/json/best_norm_params_county.json"
histories, train_true_fire, train_pred_fire, val_true_fire, val_pred_fire, y_train_true_severity, y_train_pred_severity, y_val_true_severity, y_val_pred_severity = lstm_model.train_county_models(X_train_county, X_val_county, json_file, type = "norm")


plot_perform.plot_best_model_performance(
    histories, 
    y_train_true=train_true_fire,
    y_train_pred_proba=train_pred_fire,
    y_val_true=val_true_fire,
    y_val_pred_proba=val_pred_fire,
    counties=counties,
    dir_output="model_performance/LSTM_county"
)

#______________________________________________________________________________________________________________________________
# X_train_infreq, X_val_infreq, X_test_infreq = data_setup.train_val_test_split(balanced_infrequent, test_df, counties, split_by="county")
# json_file_infreq = "model_performance/log/json/best_infreq_params_county.json"
# histories, train_true_fire, train_pred_fire, val_true_fire, val_pred_fire, y_train_true_severity, y_train_pred_severity, y_val_true_severity, y_val_pred_severity = lstm_model.train_model(X_train_infreq, X_val_infreq, json_file_infreq, type = "infreq")

# plot_perform.plot_best_model_performance(
#     histories, 
#     y_train_true=train_true_fire,
#     y_train_pred_proba=train_pred_fire,
#     y_val_true=val_true_fire,
#     y_val_pred_proba=val_pred_fire,
#     counties=counties,
#     dir_output="model_performance/LSTM_county/infreq"
# )
# #______________________________________________________________________________________________________________________________
# X_train_slide, X_val_slide, X_test_slide = data_setup.train_val_test_split(balanced_sliding, test_df, counties, split_by="county")
# json_file_slide = "model_performance/log/json/best_slide_params_county.json"
# histories, train_true_fire, train_pred_fire, val_true_fire, val_pred_fire, y_train_true_severity, y_train_pred_severity, y_val_true_severity, y_val_pred_severity = lstm_model.train_model(X_train_infreq, X_val_infreq, json_file_slide, type = "slide")

# plot_perform.plot_best_model_performance(
#     histories, 
#     y_train_true=train_true_fire,
#     y_train_pred_proba=train_pred_fire,
#     y_val_true=val_true_fire,
#     y_val_pred_proba=val_pred_fire,
#     counties=counties,
#     dir_out = "model_performance/LSTM_county/slide"
# )