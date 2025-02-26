import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.logger import logging
from src.data.data_processing import processing

TARGET = 'category'
model_name = "best_model.joblib"

def load_dataframe(path):
    """Load a DataFrame from a CSV file."""
    logging.info(f"Loading data from {path}")
    return pd.read_csv(path)

def make_X_y(dataframe: pd.DataFrame, target_column: str):
    """Split a DataFrame into feature matrix X and target vector y."""
    logging.info(f"Preparing features and target from column: {target_column}")
    df_copy = dataframe.copy()
    X = df_copy.drop(columns=[target_column], errors='ignore')
    y = df_copy[target_column]
    y = pd.get_dummies(y, drop_first=False)
    y = y.astype(int)
    return X, y

def get_predictions(model, X: pd.DataFrame):
    """Get predictions on data."""
    logging.info("Making predictions on the data")
    return model.predict(X)

def reverse_ohe(y_encoded: np.ndarray):
    """Reverse one-hot encoded array to the original categorical values."""
    logging.info("Reversing one-hot encoding to original categories")
    category_mapping = {
        0: 'Food & Dining 🍔',
        1: 'Transport 🚗',
        2: 'Shopping 🛍️',
        3: 'Utilities ⚡',
        4: 'Medical & Healthcare 🏥',
        5: 'Entertainment & Leisure 🎬',
        6: 'Rent & Housing 🏠',
        7: 'Miscellaneous 💰'
    }
    predicted_indices = np.argmax(y_encoded, axis=1)
    predicted_categories = [category_mapping[idx] for idx in predicted_indices]
    return predicted_categories

def calculate_metrics(y_actual, y_predicted):
    """Calculate various classification metrics."""
    accuracy = accuracy_score(y_actual, y_predicted)
    f1 = f1_score(y_actual, y_predicted, average='weighted')
    precision = precision_score(y_actual, y_predicted, average='weighted')
    recall = recall_score(y_actual, y_predicted, average='weighted')
    return accuracy, f1, precision, recall

def evaluate_and_log(model, X, y, dataset_name):
    """Evaluate model on given data and log results."""
    logging.info(f"Evaluating model on {dataset_name} dataset")
    y_pred = get_predictions(model, X)
    accuracy, f1, precision, recall = calculate_metrics(y, y_pred)
    
    logging.info(f"Metrics for {dataset_name} dataset:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")

def main():
    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    
    # Load the model
    logging.info(f"Loading the model from {model_name}")
    model_path = root_path / 'models' / model_name
    model = joblib.load(model_path)
    
    # Evaluate on validation set
    logging.info(f"Loading validation data from {root_path / 'data' / 'processed' / 'processed_data.csv'}")
    val_path = root_path / 'data' / 'processed' / 'processed_data.csv'
    val_data = load_dataframe(val_path)
    X_val, y_val = make_X_y(val_data, TARGET)
    evaluate_and_log(model, X_val, y_val, "Validation")
    
    # Make predictions on test set (no labels)
    logging.info(f"Loading test data from {root_path / 'data' / 'raw' / 'Expense_Dataset.csv'}")
    test_path = root_path / 'data' / 'raw' / 'Expense_Dataset.csv'
    test_data = load_dataframe(test_path)
    X_test = test_data.drop(columns=['category'], errors='ignore')
    X_test = processing(X_test)
    
    raw_test = load_dataframe(root_path / 'data' / 'raw' / 'Expense_Dataset.csv')
    predictions = get_predictions(model, X_test)
    predictions_original = reverse_ohe(predictions)
    
    # Create the submission DataFrame
    logging.info("Creating the submission DataFrame")
    submission_df = pd.DataFrame({'id': raw_test['title'], 'Target': predictions_original})
    
    # Create submission directory if it doesn't exist
    submission_path = root_path / 'data' / 'submission'
    submission_path.mkdir(exist_ok=True)
    
    # Save the submission file to a CSV
    submission_file = submission_path / 'submission.csv'
    submission_df.to_csv(submission_file, index=False)
    
    logging.info(f"Submission file created successfully at {submission_file}")

if __name__ == "__main__":
    logging.info("Starting prediction process...")
    main()
