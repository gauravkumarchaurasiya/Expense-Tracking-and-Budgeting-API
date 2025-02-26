# from fastapi import FastAPI, HTTPException, Depends
# # from motor.motor_asyncio import AsyncIOMotorClient
# import uvicorn
# import joblib
# import numpy as np
# from pydantic import BaseModel
# from typing import List
# import os
# import pandas as pd
# from pymongo import MongoClient
# from bson.objectid import ObjectId
# from dotenv import load_dotenv
# from pathlib import Path 
# from gensim.models import Word2Vec 
# # from src.data.data_processing import processing
# from src.model.predict import reverse_ohe,get_predictions,processing


# app = FastAPI()

# # MongoDB Connection
# # load_dotenv ()
# # MONGODB_URI = os.environ['MONGODB_URI']

# # client = MongoClient(MONGODB_URI)

# # db = client.Blog
# # posts_collection = db.Posts

# # for db_name in client.list_database_names():
# #     print(db_name)

# # Load Word2Vec Model
# WORD2VEC_MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / "word2vec.model"
# word2vec_model = Word2Vec.load(str(WORD2VEC_MODEL_PATH))  # Load Word2Vec model

# # Load Trained Classifier Model
# MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / "best_model.joblib"
# with open(MODEL_PATH, "rb") as model_file:
#     model = joblib.load(model_file)
#     print("Classifier model loaded successfully")

# # Pydantic Models
# class Expense(BaseModel):
#     title: str
#     amount: float
#     account_encoded: str
#     type: str
#     category: str

# class ExpenseCreate(BaseModel):
#     title: str
#     account: str
#     amount_encoded: float
#     type_encoded: str

# class PredictionInput(BaseModel):
#     title: str
#     amount: float
#     account_encoded: int
#     type_encoded: int

# # CRUD Operations
# @app.post("/expenses/", response_model=Expense)
# async def create_expense(expense: Expense):
#     new_expense = await db.Posts.insert_one(expense.dict())
#     return {"id": str(new_expense.inserted_id), **expense.dict()}

# @app.get("/expenses/", response_model=List[Expense])
# async def get_expenses():
#     expenses = await db.Posts.find().to_list(100)
#     return expenses

# # @app.post("/predict/")
# # async def predict_category(data: PredictionInput):
# #     # input_data = np.array([[data.amount]])  
# #     # prediction = model.predict(input_data)[0]
# #     # return {"predicted_category": prediction}
# #     input_data = pd.DataFrame([data.dict()])
# #     input_data_processed = processing(input_data)
# #     prediction = get_predictions(model, input_data_processed)
# #     prediction_original = reverse_ohe(prediction)
# #     # prediction = model.predict(input_data)[0]
# #     return {"predicted_category": prediction_original}

# # Predict Category Endpoint
# @app.post("/predict/")
# async def predict_category(data: PredictionInput):
#     try:
#         # Convert the incoming data to a DataFrame
#         input_data = pd.DataFrame([data.dict()])
#         print(f"Input data: {input_data}")  # Debugging

#         # Apply data processing (including using Word2Vec for text processing)
#         input_data_processed = processing(input_data, word2vec_model)
#         print(f"Processed data: {input_data_processed}")  # Debugging

#         # Get prediction from the model
#         prediction = get_predictions(model, input_data_processed)
#         print(f"Prediction: {prediction}")  # Debugging

#         # Reverse transformations (e.g., reverse one-hot encoding)
#         prediction_original = reverse_ohe(prediction)
#         print(f"Reversed Prediction: {prediction_original}")  # Debugging

#         return {"predicted_category": prediction_original}

#     except Exception as e:
#         # Handle any exceptions and provide more context
#         print(f"Error in prediction: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error during prediction")


# # Run FastAPI
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI
from src.backend.routes import router
import uvicorn

app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
