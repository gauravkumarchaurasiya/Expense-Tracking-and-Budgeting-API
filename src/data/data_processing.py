import logging
import spacy
import pandas as pd
import numpy as np
from pathlib import Path
from src.logger import logging
from gensim.models import Word2Vec
from src.data import install_dependancy
from sklearn.feature_extraction.text import TfidfVectorizer
from src.model.model import save_model
def load_raw_data(input_path: Path) -> pd.DataFrame:
    """Load the raw data from the CSV file."""
    logging.info(f"Loading raw data from {input_path}...")
    raw_data = pd.read_csv(input_path)
    rows, columns = raw_data.shape
    logging.info(f"{input_path.stem} data read having {rows} rows and {columns} columns")
    return raw_data

def install_spacy_dependancy():
    """Check and install the spaCy model if not already installed."""
    # Check if the 'en_core_web_sm' model is already installed
    logging.info("Checking for spaCy model 'en_core_web_sm'...")
    if "en_core_web_sm" not in spacy.util.get_installed_models():
        logging.info("spaCy model not found. Installing 'en_core_web_sm' model...")
        install_dependancy.download_spacy_model()  # Download the model if not installed
    else:
        logging.info("spaCy model 'en_core_web_sm' is already installed.")

# Function to preprocess text using spaCy
def preprocess_text_spacy(text):
    """Preprocess the text using spaCy by lemmatizing and removing stopwords."""
    nlp = spacy.load("en_core_web_sm")  # Load the model
    logging.info("Preprocessing text data with spaCy...")
    doc = nlp(str(text).lower())  # Convert to lowercase and process with spaCy
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]  # Lemmatization & stopword removal
    return " ".join(tokens)

def train_word2vec_model(sentences):
    """Train a Word2Vec model on the sentences."""
    logging.info("Training Word2Vec model...")
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    model.save(str(Path(__file__).parent.parent.parent/'models'/'word2vec.model'))  # Save the model for future use
    return model


def vectorize_text_using_word2vec(df: pd.DataFrame, model: Word2Vec) -> pd.DataFrame:
    """Vectorize text data using Word2Vec embeddings."""
    logging.info("Vectorizing text using Word2Vec...")

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

    logging.info(f"Word2Vec vectorization completed. Dataframe shape: {df.shape}")
    return df

def processing(df: pd.DataFrame) -> pd.DataFrame:
    """Process data by preprocessing and vectorizing using Word2Vec."""
    logging.info("Step 2: Preprocessing the 'title' column")
    df["clean_title"] = df["title"].apply(preprocess_text_spacy)

    # Display the first few cleaned titles
    logging.info(f"Cleaned titles:\n{df[['title', 'clean_title']].head()}")

    # Train Word2Vec model on the entire 'clean_title' column
    sentences = df["clean_title"].tolist()
    model = train_word2vec_model(sentences)

    # Vectorize the data
    logging.info("Step 3: Starting vectorization process")
    df = vectorize_text_using_word2vec(df, model)

    # Handle missing data
    logging.info("Removing missing data...")
    df = df.dropna()

    return df
    
def save_data(data: pd.DataFrame, output_path: Path):
    """Save the processed data to a CSV file."""
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / "processed_data.csv"
    data.to_csv(output_file_path, index=False)
    logging.info(f"Data saved successfully to {output_file_path}")

def main():
    """Main function to run the data processing pipeline."""
    logging.info("Starting the data processing pipeline...")

    current_path = Path(__file__)
    root_path = current_path.parent.parent.parent
    raw_data_path = root_path / 'data' / 'raw'
    
    logging.info("Step 1: Data extraction from CSV file")
    
    input_file_name = "Expense_Dataset.csv"
    
    # Load raw data
    df = load_raw_data(raw_data_path / input_file_name)
    
    df = processing(df)
    save_data(df,root_path/'data'/'processed')
    
    logging.info("Data processing pipeline completed successfully.")

if __name__ == '__main__':
    main()
