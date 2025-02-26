import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from src.logger import logging
from src.data.data_processing import preprocess_text_spacy
from src.model.model import save_model
from gensim.models import Word2Vec  # Import Word2Vec

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

def train_word2vec_model(sentences):
    """Train a Word2Vec model on the sentences."""
    logging.info("Training Word2Vec model...")
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    model.save('word2vec.model')  # Save the model for future use
    return model

def vectorization_with_word2vec(df: pd.DataFrame, model: Word2Vec) -> pd.DataFrame:
    """Vectorize using Word2Vec."""
    logging.info("Word2Vec vectorization started...")

    def get_average_word_vector(tokens):
        """Convert a list of tokens into an average word vector."""
        word_vectors = [model.wv[token] for token in tokens if token in model.wv]
        if len(word_vectors) == 0:
            return [0] * model.vector_size  # Return a zero vector if no valid word vectors
        return np.mean(word_vectors, axis=0)  # Return the average vector

    # Apply Word2Vec to the 'clean_title' column
    wordvecs = df["clean_title"].apply(get_average_word_vector)
    wordvecs_df = pd.DataFrame(wordvecs.tolist(), columns=[f"wordvec_{i}" for i in range(model.vector_size)])

    # Merge Word2Vec features with original dataset
    df = pd.concat([df, wordvecs_df], axis=1)

    # Drop raw text columns (optional)
    df.drop(["title", "clean_title"], axis=1, inplace=True)

    logging.info("Word2Vec vectorization completed successfully.")
    # Display the new dataset with Word2Vec features
    logging.info(f"Word2Vec DataFrame: \n{df.head()}")
    return df

def processing(df: pd.DataFrame, model: Word2Vec) -> pd.DataFrame:
    # Apply preprocessing to the "title" column
    logging.info("Step 2: Preprocessing the 'title' column")
    df["clean_title"] = df["title"].apply(preprocess_text_spacy)

    # Display the first few cleaned titles
    logging.info(f"Cleaned titles:\n{df[['title', 'clean_title']].head()}")

    # Perform Word2Vec vectorization
    logging.info("Step 3: Starting Word2Vec vectorization process")
    df = vectorization_with_word2vec(df, model)

    # Handle missing data
    logging.info("Removing missing data...")
    df = df.dropna()

    return df

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

    # Load pre-trained Word2Vec model
    logging.info(f"Loading the Word2Vec model from 'word2vec.model'")
    word2vec_model = Word2Vec.load(str(root_path/'models'/'word2vec.model'))

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
    X_test = processing(X_test, word2vec_model)

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
