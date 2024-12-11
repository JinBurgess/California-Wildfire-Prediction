from main import *
from lib.model_prediction import WildfirePredictor

fire_prediction = WildfirePredictor()

# Base Approach
#______________________________________________________________________________________________________________________________
# X_train_city, X_val_city, X_test_city = data_setup.train_val_test_split(train_df, test_df, cities, split_by="name")

# json_file = "model_performance/log/json/best_norm_params.json"
# result = fire_prediction.predict_func(X_test_city, json_file, save = "model_predictions/norm")
# plot_perform.plot_result_df(result, type = "norm", dir_out= "model_performance/predictions")

# Minority Weight Approach
#______________________________________________________________________________________________________________________________
# json_file_minor_weight = "model_performance/log/json/best_minor_weight_params.json"
# result = fire_prediction.predict_func(X_test_city, json_file_minor_weight, save = "model_predictions/minor_weight", minor_weight = True)
# plot_perform.plot_result_df(result, type = "minor_weight", dir_out= "model_performance/predictions")

# Infrequent Sampling Approach
#______________________________________________________________________________________________________________________________
# balanced_infrequent = balancing.infrequent_over_sample(cities, train_df, by = "name")
# X_train_infreq, X_val_infreq, X_test_infreq = data_setup.train_val_test_split(balanced_infrequent, test_df, cities, split_by="name")
# json_file_infreq = "model_performance/log/json/best_infreq_params.json"
# result = fire_prediction.predict_func(X_test_infreq, json_file_infreq, save = "model_predictions/infreq", minor_weight = False)
# plot_perform.plot_result_df(result, type = "infreq", dir_out= "model_performance/predictions")

# Sliding Window Approach
#______________________________________________________________________________________________________________________________
balanced_slide = balancing.sliding_over_sample(cities, train_df, by = "name")
X_train_slide, X_val_slide, X_test_slide = data_setup.train_val_test_split(balanced_slide, test_df, cities, split_by="name")
json_file_slide = "model_performance/log/json/best_slide_params.json"
result = fire_prediction.predict_func(X_test_slide, json_file_slide, save = "model_predictions/slide", minor_weight = False)
plot_perform.plot_result_df(result, type = "slide", dir_out= "model_performance/predictions")

# Regional Grouping Approach
#______________________________________________________________________________________________________________________________s