import logging
import spacy
import pandas as pd
from pathlib import Path
from src.logger import logging
from src.data import install_dependancy
from sklearn.feature_extraction.text import TfidfVectorizer

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

def vectorization(df: pd.DataFrame) -> pd.DataFrame:
    """Perform TF-IDF vectorization on the 'clean_title' column."""
    logging.info("TF-IDF vectorization started ...")
    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=100)  # Extract top 100 important words
    X_tfidf = vectorizer.fit_transform(df["clean_title"]).toarray()
    
    # Convert to DataFrame
    tfidf_df = pd.DataFrame(X_tfidf, columns=vectorizer.get_feature_names_out())
    
    # Merge TF-IDF features with original dataset
    df = pd.concat([df, tfidf_df], axis=1)

    # Drop raw text columns (optional)
    df.drop(["title", "clean_title"], axis=1, inplace=True)

    logging.info("TF-IDF vectorization completed successfully.")
    # Display the new dataset with TF-IDF features
    logging.info(f"TF-IDF DataFrame: \n{df.head()}")
    return df
def processing(df:pd.DataFrame)->pd.DataFrame:
    # Apply preprocessing to the "title" column
    logging.info("Step 2: Preprocessing the 'title' column")
    df["clean_title"] = df["title"].apply(preprocess_text_spacy)

    # Display the first few cleaned titles
    logging.info(f"Cleaned titles:\n{df[['title', 'clean_title']].head()}")

    # Perform vectorization
    logging.info("Step 3: Starting vectorization process")
    df = vectorization(df)
    
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

    logging.info("Data processing pipeline completed successfully.")

if __name__ == '__main__':
    main()
