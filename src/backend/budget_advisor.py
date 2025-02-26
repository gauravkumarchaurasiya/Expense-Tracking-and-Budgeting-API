import google.generativeai as genai
import os
from dotenv import load_dotenv
from src.backend.database import expenses_collection, budgets_collection
import pandas as pd
import re  # For cleaning category names
from datetime import datetime

load_dotenv()

# Setup Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# Budget Categories & Percentage Allocations
CATEGORIES = [
    "Food & Dining", "Transport", "Shopping", "Utilities",
    "Medical & Healthcare", "Entertainment & Leisure",
    "Rent & Housing", "Miscellaneous"
]

BUDGET_PERCENTAGES = {
    "Food & Dining": 0.20,
    "Transport": 0.10,
    "Shopping": 0.10,
    "Utilities": 0.15,
    "Medical & Healthcare": 0.12,
    "Entertainment & Leisure": 0.08,
    "Rent & Housing": 0.20,
    "Miscellaneous": 0.05
}

def fetch_budget_from_db():
    """Fetch total budget from DB and distribute it into categories."""
    budget_data = budgets_collection.find_one({}, {"_id": 0, "monthly_budget": 1})
    
    if not budget_data:
        print("⚠️ No budget data found in DB. Using default 0 budget.")
        return {cat: 0 for cat in CATEGORIES}  # Default all budgets to 0
    
    total_budget = budget_data.get("monthly_budget", 0)
    
    # Allocate budget dynamically using percentage distribution
    category_budgets = {cat: total_budget * BUDGET_PERCENTAGES[cat] for cat in CATEGORIES}
    
    print("✅ Budget Data from DB:", category_budgets)
    return category_budgets


def clean_category_name(category):
    """Remove emojis and return a clean category name."""
    if isinstance(category, list):  
        category = category[0]  # Extract first item if it's a list
    return re.sub(r'[^\w\s&]', '', category).strip()  # Remove emojis/special characters


def analyze_budget(year=None, month=None):
    """Fetch transactions for a given month and analyze budget."""

    if not year or not month:
        now = datetime.now()
        year, month = now.year, now.month

    # Define the start and end dates
    start_date = datetime(year, month, 1).isoformat()  # "2025-02-01T00:00:00"
    end_month = month + 1 if month < 12 else 1
    end_year = year if month < 12 else year + 1
    end_date = datetime(end_year, end_month, 1).isoformat()  # "2025-03-01T00:00:00"

    # Query transactions correctly
    transactions = list(expenses_collection.find({
        "date": {
            "$gte": start_date,  # Start of the month in ISO format
            "$lt": end_date      # Start of next month in ISO format
        }
    }, {"_id": 0, "amount": 1, "category": 1, "date": 1}))

    if not transactions:
        return f"⚠️ No transactions found for {month}/{year}!"

    # Convert transactions to DataFrame
    df = pd.DataFrame(transactions)
    df["category"] = df["category"].apply(clean_category_name)

    # Fetch the budget from DB
    category_budgets = fetch_budget_from_db()
    category_totals = df.groupby("category")["amount"].sum().to_dict()

    # Analyze budget
    analysis = {}
    for category in CATEGORIES:
        spent = category_totals.get(category, 0)
        budget = category_budgets.get(category, 0)
        status = "Within Budget" if spent <= budget else "Over Budget"
        analysis[category] = {"spent": spent, "budget": budget, "status": status}

    return {"year": year, "month": month, "analysis": analysis}


def generate_budget_advice(year=None, month=None):
    """Generates short budget advice based on spending trends for a specific month."""
    
    analysis_data = analyze_budget(year, month)
    if isinstance(analysis_data, str):
        return analysis_data  # Return error message if no transactions found
    
    year, month, analysis = analysis_data["year"], analysis_data["month"], analysis_data["analysis"]
    
    prompt = f"""
    You are an AI financial advisor analyzing expenses for {month}/{year}. 

    - If spending is **over budget**, suggest practical ways to reduce expenses.
    - If spending is **too low**, recommend whether it's okay to increase it.
    - Provide **actionable and specific** advice based on spending trends.

    Budget Analysis:
    {analysis}

    Your response should be **brief (2-3 sentences), practical, and helpful**.
    """
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating advice: {str(e)}"
    
def validate_category_with_gemini(title: str, amount: float, account: str, type_: str, predicted_category: str):
    """Uses Gemini API to validate the ML-predicted category and provide corrections if necessary."""
    prompt = f"""
    You are an AI financial assistant validating an expense category. 
    
    **Given the following details:**
    - **Title:** {title}
    - **Amount:** ₹{amount}
    - **Account Type:** {account}
    - **Expense Type:** {type_}
    - **Predicted Category by ML:** {predicted_category}

    Confirm whether this category is correct. If incorrect, suggest a **better category** from this list:
    {", ".join(CATEGORIES)}

    **Respond ONLY with the category name** (no explanations).
    """

    try:
        response = model.generate_content(prompt)
        gemini_category = response.text.strip()

        # Validate Gemini's response
        if gemini_category in CATEGORIES:
            return gemini_category  # Use Gemini's category if valid
        return predicted_category  # Fallback to ML model's category if Gemini fails
    except Exception as e:
        print(f"⚠️ Gemini API error: {str(e)}")
        return predicted_category  
    
if __name__ == "__main__":
    # You can specify the month/year or leave it blank to use the current month
    print(generate_budget_advice(2025, 2))  # Example for February 2025
