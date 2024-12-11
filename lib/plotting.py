from common_imports import *

class PerformanceModeling:
    def __init__(self, sequence_length=15, threshold=0.8, patience=10):
        """
        Initialize PerformanceModeling with default parameters.
        
        Parameters:
        - sequence_length: Length of the input sequences for LSTM models.
        - threshold: Threshold for binary classification (fire prediction).
        - patience: Early stopping patience for model training.
        """
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.patience = patience

    @staticmethod
    def plot_performance(history, y_train_true, y_train_pred_proba, y_val_true, y_val_pred_proba, dir_out="model_performance"):
        """
        Plots training/validation loss, accuracy, F1-score, and ROC curves.
        """
        history_dict = history.history

        # Handle missing keys with default empty lists
        loss = history_dict.get('loss', [])
        val_loss = history_dict.get('val_loss', [])
        accuracy = history_dict.get('fire_output_accuracy', [])
        val_accuracy = history_dict.get('val_fire_output_accuracy', [])
        f1_score = history_dict.get('fire_output_f1_score', [])
        val_f1_score = history_dict.get('val_fire_output_f1_score', [])

        # Compute ROC curves and AUC
        fpr_train, tpr_train, _ = roc_curve(y_train_true, y_train_pred_proba)
        roc_auc_train = auc(fpr_train, tpr_train)
        
        fpr_val, tpr_val, _ = roc_curve(y_val_true, y_val_pred_proba)
        roc_auc_val = auc(fpr_val, tpr_val)

        # Create subplots
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))

        # Loss Plot
        ax[0, 0].plot(loss, label='Training Loss', color='blue')
        ax[0, 0].plot(val_loss, label='Validation Loss', color='orange')
        ax[0, 0].set_title('Loss Over Epochs')
        ax[0, 0].legend()
        ax[0, 0].grid(True)

        # Accuracy Plot
        ax[0, 1].plot(accuracy, label='Training Accuracy', color='green')
        ax[0, 1].plot(val_accuracy, label='Validation Accuracy', color='red')
        ax[0, 1].set_title('Accuracy Over Epochs')
        ax[0, 1].legend()
        ax[0, 1].grid(True)

        # ROC Curve
        ax[1, 0].plot(fpr_train, tpr_train, label=f'Training ROC (AUC = {roc_auc_train:.2f})', color='blue')
        ax[1, 0].plot(fpr_val, tpr_val, label=f'Validation ROC (AUC = {roc_auc_val:.2f})', color='orange')
        ax[1, 0].plot([0, 1], [0, 1], 'k--')
        ax[1, 0].set_title('ROC Curve')
        ax[1, 0].legend()
        ax[1, 0].grid(True)

        # F1 Score
        ax[1, 1].plot(f1_score, label='Training F1', color='green')
        ax[1, 1].plot(val_f1_score, label='Validation F1', color='red')
        ax[1, 1].set_title('F1 Score Over Epochs')
        ax[1, 1].legend()
        ax[1, 1].grid(True)

        # Save and show the plot
        os.makedirs('model_performance/LSTM', exist_ok=True)
        file_name = f'{dir_out}.png'
        plt.tight_layout()
        fig.savefig(file_name, dpi=300, format='png')
        plt.show()

    @staticmethod
    def plot_result_df(df, type = "norm", dir_out = ""):
        """
        Plots prediction performance histograms and ROC curves.
        """
        fpr, tpr, _ = roc_curve(df['FireTrue'], df['FirePredictionBinary'])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(14, 10))

        # Fire Prediction vs True Fire
        plt.subplot(2, 2, 1)
        plt.hist(df['FireTrue'], bins=30, alpha=0.7, label='True Fire', color='blue')
        plt.hist(df['FirePrediction'], bins=30, alpha=0.7, label='Predicted Fire', color='orange')
        plt.title('Fire Prediction vs True Fire Distribution')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.hist(df['FireTrue'], bins=30, alpha=0.7, label='True Fire', color='blue')
        plt.hist(df['FirePredictionBinary'], bins=30, alpha=0.7, label='Predicted Fire', color='orange')
        plt.title('Fire Prediction Binary vs True Fire Distribution')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)

        # Severity Prediction vs True Severity
        plt.subplot(2, 2, 3)
        plt.hist(df['SeverityTrue'], bins=30, alpha=0.7, label='True Severity', color='blue')
        plt.hist(df['SeverityPrediction'], bins=30, alpha=0.7, label='Predicted Severity', color='orange')
        plt.title('Severity Prediction vs True Severity Distribution')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)

        # ROC Curve
        plt.subplot(2, 2, 4)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')

        plt.tight_layout()
        os.makedirs(f'{dir_out}', exist_ok=True)
        plt.savefig(f'{dir_out}/{type}.png', dpi=300, format='png')
        plt.show()

    @staticmethod
    def plot_best_model_performance(histories, y_train_true, y_train_pred_proba, y_val_true, y_val_pred_proba, counties, dir_output =""):
        """
        Plots the performance metrics of the model training and validation,
        including loss, accuracy, and ROC curves for each county.
        
        Parameters:
        - histories: Dictionary of training history metrics for each county.
        - counties: List of county names to plot performance for.
        - threshold: Threshold for binary classification.
        """

        # Use the dictionary for plotting
        for idx, county in enumerate(counties):
            if county not in histories:
                print(f"No history data for county: {county}")
                continue

            metrics = histories[county]
            print(f"Processing data for county: {county}")
    
            # Compute ROC curves and AUC for training and validation
            fpr_train, tpr_train, _ = roc_curve(y_train_true[idx], y_train_pred_proba[idx])
            roc_auc_train = auc(fpr_train, tpr_train)
            
            fpr_val, tpr_val, _ = roc_curve(y_val_true[idx], y_val_pred_proba[idx])
            roc_auc_val = auc(fpr_val, tpr_val)

            # Plotting the metrics for each county
            fig, ax = plt.subplots(2, 2, figsize=(12, 10))

            # Loss Plot
            ax[0, 0].plot(metrics['loss'], label='Training Loss', color='blue')
            ax[0, 0].plot(metrics['val_loss'], label='Validation Loss', color='orange')
            ax[0, 0].set_title(f'{county} - Training and Validation Loss Over Epochs')
            ax[0, 0].set_xlabel('Epochs')
            ax[0, 0].set_ylabel('Loss')
            ax[0, 0].legend()
            ax[0, 0].grid(True)

            # Accuracy Plot
            ax[0, 1].plot(metrics['fire_output_accuracy'], label='Training Accuracy', color='green')
            ax[0, 1].plot(metrics['val_fire_output_accuracy'], label='Validation Accuracy', color='red')
            ax[0, 1].set_title(f'{county} - Training and Validation Accuracy Over Epochs')
            ax[0, 1].set_xlabel('Epochs')
            ax[0, 1].set_ylabel('Accuracy')
            ax[0, 1].legend()
            ax[0, 1].grid(True)

            # ROC Curve Plot (placeholders; real values needed)
            if fpr_train.size > 0 and tpr_train.size > 0:
                ax[1, 0].plot(fpr_train, tpr_train, label=f'Training ROC (AUC = {roc_auc_train:.2f})', color='blue')
                ax[1, 0].plot(fpr_val, tpr_val, label=f'Validation ROC (AUC = {roc_auc_val:.2f})', color='orange')
                ax[1, 0].plot([0, 1], [0, 1], 'k--', label='Chance Level')
                ax[1, 0].set_title(f'{county} - ROC Curve')
                ax[1, 0].set_xlabel('False Positive Rate')
                ax[1, 0].set_ylabel('True Positive Rate')
                ax[1, 0].legend()
                ax[1, 0].grid(True)
            else:
                ax[1, 0].axis('off')  # Hide plot if data is unavailable

            # F1 Score Plot
            if 'fire_output_f1_score' in metrics:
                ax[1, 1].plot(metrics['fire_output_f1_score'], label='Training F1 Score', color='green')
                ax[1, 1].plot(metrics['val_fire_output_f1_score'], label='Validation F1 Score', color='red')
                ax[1, 1].set_title(f'{county} - Training and Validation F1 Over Epochs')
                ax[1, 1].set_xlabel('Epochs')
                ax[1, 1].set_ylabel('F1 Score')
                ax[1, 1].legend()
                ax[1, 1].grid(True)
            else:
                ax[1, 1].axis('off')  # Hide plot if F1 data is unavailable


            plt.tight_layout()
            os.makedirs(f'{dir_output}', exist_ok=True)
            fig.savefig(f'{dir_output}/{county}.png', dpi=300, format='png')
            plt.show()
