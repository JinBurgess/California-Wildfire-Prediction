from common_imports import *

class CustomLoss:
    def __init__(self, gamma=2.0, alpha=0.25, fire_threshold=0.8, patience=10):
        """
        Initialize the class with parameters for focal loss, thresholding, and early stopping.
        
        Parameters:
        - gamma: Focal loss parameter to reduce easy example weight.
        - alpha: Focal loss parameter for class weighting.
        - fire_threshold: Threshold for fire prediction binary classification.
        - patience: Early stopping patience for model training.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.fire_threshold = fire_threshold
        self.patience = patience

    @staticmethod
    def dynamic_weighted_binary_crossentropy(y_true, y_pred):
        # Ensure y_true is float32 to match y_pred
        y_true = tf.cast(y_true, tf.float32)
        
        # Calculate the proportion of positive and negative samples in the batch
        positive_count = tf.reduce_sum(y_true)  # Number of positives
        negative_count = tf.reduce_sum(1.0 - y_true)  # Number of negatives
        
        # Avoid division by zero
        total_count = positive_count + negative_count
        positive_weight = negative_count / total_count  # Weight for positives
        negative_weight = positive_count / total_count  # Weight for negatives
        
        # Assign weights dynamically
        weights = tf.where(y_true == 1, positive_weight, negative_weight)
        
        # Compute binary crossentropy
        loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Apply weights and compute the mean
        return tf.reduce_mean(weights * loss)

    @staticmethod
    def weighted_binary_crossentropy(y_true, y_pred):
        weights = tf.where(y_true == 1, 2.0, 1.0)  # Weight minority class higher
        return tf.reduce_mean(weights * tf.keras.losses.binary_crossentropy(y_true, y_pred))
    
    # def focal_loss(self, y_true, y_pred):
    #     """
    #     Focal loss for handling class imbalance with tunable gamma and alpha.
    #     """
    #     pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    #     loss = -self.alpha * (1 - pt) ** self.gamma * tf.math.log(pt + tf.keras.backend.epsilon())
    #     return tf.reduce_mean(loss)

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Compute the F1 score based on true and predicted labels.
        """
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(tf.round(y_pred), 'float32')
        precision = tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_pred) + tf.keras.backend.epsilon())
        recall = tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_true) + tf.keras.backend.epsilon())
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
