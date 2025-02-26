import joblib
import pandas as pd
from gensim.models import Word2Vec
from pathlib import Path
from src.model.predict import reverse_ohe, get_predictions, processing

# Load Models
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "best_model.joblib"
WORD2VEC_PATH = Path(__file__).parent.parent.parent / "models" / "word2vec.model"

model = joblib.load(MODEL_PATH)
word2vec_model = Word2Vec.load(str(WORD2VEC_PATH))

ACCOUNT_MAPPING = {"Cash": 0, "Bank": 1, "Credit Card": 2}  # Extend as needed
TYPE_MAPPING = {"Expense": 0, "Income": 1}

def predict_category(title: str, amount: float, account: str, type: str):
    
    try:
        # ✅ **Convert `account` & `type` to numerical values**
        account_encoded = ACCOUNT_MAPPING.get(account, -1)
        type_encoded = TYPE_MAPPING.get(type, -1)

        if account_encoded == -1 or type_encoded == -1:
            raise ValueError("Invalid account or type value provided to ML model")
        
        input_data = pd.DataFrame([{
            "title": title,
            "amount": amount,
            "account_encoded": account_encoded,
            "type_encoded": type_encoded
    }])
    except:
        print(f"Error in predict_category: {e}")
        return "Unknown"

    processed_data = processing(input_data, word2vec_model)
    prediction = get_predictions(model, processed_data)
    return reverse_ohe(prediction)  # Return predicted category
