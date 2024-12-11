from main import *
from lib.lstm_model_trainer import ModelTrainer

lstm_model = ModelTrainer()

# Load 
#______________________________________________________________________________________________________________________________
# X_train_city, X_val_city, X_test_city = data_setup.train_val_test_split(train_df, test_df, cities, split_by="name")

# results_norm = lstm_model.randomized_cv_search(X_train_city, cities, param_grid, n_iter=10, cv_folds= 3)
# best_model_norm = lstm_blocks.return_best_result(results_norm)

# minority_weight data handling
#______________________________________________________________________________________________________________________________
# X_train_city, X_val_city, X_test_city = data_setup.train_val_test_split(train_df, test_df, cities, split_by="name")
# X_norm = pd.concat([X_train_city, X_val_city], axis=0)
# results_norm_weight = lstm_model.randomized_cv_search(X_norm, cities, param_grid, n_iter=10, cv_folds= 5, minor_weight = True)
# best_model_norm_weight = lstm_blocks.return_best_result(results_norm_weight, type= "minor_weight")

# infrequent data handling
#______________________________________________________________________________________________________________________________
# balanced_infrequent = balancing.infrequent_over_sample(cities, train_df, by = "name")
# X_train_infreq, X_val_infreq, X_test_infreq = data_setup.train_val_test_split(balanced_infrequent, test_df, cities, split_by="name")

# results_infreq = lstm_model.randomized_cv_search(X_train_infreq, cities, param_grid, n_iter=10, cv_folds= 3)
# best_model_infreq = lstm_blocks.return_best_result(results_infreq, type="infreq")

# sliding window data handling
#______________________________________________________________________________________________________________________________
# balanced_sliding = balancing.sliding_over_sample(cities, train_df, by = "name")
# X_train_slide, X_val_slide, X_test_slide = data_setup.train_val_test_split(balanced_sliding, test_df, cities, split_by="name")

# X_slide = pd.concat([X_train_slide, X_val_slide], axis=0)
# results_slide = lstm_model.randomized_cv_search(X_slide, cities, param_grid, n_iter=10, cv_folds= 3)
# best_model_slide = lstm_blocks.return_best_result(results_slide, type="slide")