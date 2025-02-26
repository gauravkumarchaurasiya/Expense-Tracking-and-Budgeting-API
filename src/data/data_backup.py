import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# MongoDB Connection
MONGODB_URI = os.getenv("MONGODB_URI")

# Connect to MongoDB
client = MongoClient(MONGODB_URI)
db = client["ExpenseDB"]
expenses_collection = db["expenses"]
budgets_collection = db["budgets"]

# Fetch data from MongoDB
data = list(expenses_collection.find())

if data:
    # Convert MongoDB ObjectId to string
    for item in data:
        item["_id"] = str(item["_id"])

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Create 'raw' folder if it doesn't exist
    root_path = Path(__file__).parent.parent.parent
    raw_folder = root_path/'data'/"raw"
    os.makedirs(raw_folder, exist_ok=True)

    # Save to CSV
    csv_path = raw_folder/ "Expense_Dataset.csv"
    df.to_csv(csv_path, index=False)

    print(f"✅ Data successfully saved to {csv_path}")
else:
    print("⚠️ No data found in the collection.")

# Close MongoDB connection
client.close()
