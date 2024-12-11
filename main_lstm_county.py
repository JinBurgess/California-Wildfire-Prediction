from main import *

from lib.lstm_model_county import CountyModelTrainer
from lib.plotting import PerformanceModeling

lstm_model = CountyModelTrainer()
plotting_methods = PerformanceModeling()

#______________________________________________________________________________________________________________________________
# Load Data
X_train_county, X_val_county, X_test_county = data_setup.train_val_test_split(train_df, test_df, counties, split_by="county")

X_norm = pd.concat([X_train_county, X_val_county], axis=0)
results_norm = lstm_model.randomized_cv_search(X_norm, counties, param_grid, n_iter=10, cv_folds= 3)
best_result_norm = lstm_blocks.return_best_result_county(results_norm, counties, type = "norm")

# results_infreq = model_train.randomized_cv_search(X_train_infreq, X_val, counties, param_grid, n_iter=3)
# results_slide = model_train.randomized_cv_search(X_train_slide, X_val, counties, param_grid, n_iter=3)

# infrequent data handling
#______________________________________________________________________________________________________________________________
# balanced_infrequent = balancing.infrequent_over_sample(counties, train_df, by = "county")
# X_train_infreq, X_val_infreq, X_test_infreq = data_setup.train_val_test_split(balanced_infrequent, test_df, counties, split_by="county")

# X_infreq = pd.concat([X_train_infreq, X_val_infreq], axis=0)
# results_infreq = lstm_model.randomized_cv_search(X_infreq, counties, param_grid, n_iter=10, cv_folds= 3)
# best_model_infreq = lstm_blocks.return_best_result(results_infreq, type="infreq")

# sliding window data handling
#______________________________________________________________________________________________________________________________
# balanced_sliding = balancing.sliding_over_sample(counties, train_df, by = "county")
# X_train_slide, X_val_slide, X_test_slide = data_setup.train_val_test_split(balanced_sliding, test_df, counties, split_by="county")

# X_slide = pd.concat([X_train_slide, X_val_slide], axis=0)
# results_slide = lstm_model.randomized_cv_search(X_slide, counties, param_grid, n_iter=10, cv_folds= 3)
# best_model_slide = lstm_blocks.return_best_result(results_slide, type="slide")