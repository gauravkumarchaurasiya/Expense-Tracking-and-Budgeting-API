import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

client = MongoClient(MONGODB_URI)
db = client["ExpenseDB"]
expenses_collection = db["expenses"]
budgets_collection = db["budgets"]
