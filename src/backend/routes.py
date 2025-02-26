from fastapi import APIRouter, HTTPException
from src.backend.schemas import *
from src.backend.database import expenses_collection, budgets_collection
from datetime import datetime
from src.backend.ml_model import predict_category
import pandas as pd
from datetime import datetime, timedelta
from src.backend.budget_advisor import *

router = APIRouter()

# 📌 Add Expense with Category Prediction & Save to DB
@router.post("/expenses/")
async def add_expense(expense: ExpenseCreate):
    try:
        # Predict category
        predicted_category = predict_category(expense.title, expense.amount, expense.account, expense.type)

         # Step 2: Validate category using Gemini API
        final_category = validate_category_with_gemini(expense.title, expense.amount, expense.account, expense.type, predicted_category)

        # Convert to dictionary & add category and date
        expense_dict = expense.dict()
        expense_dict["category"] = final_category
        expense_dict["date"] = datetime.now().isoformat()

        # Insert into DB
        inserted_expense = expenses_collection.insert_one(expense_dict)
        expense_dict["_id"] = str(inserted_expense.inserted_id)  # Convert ObjectId to string
        
        return expense_dict
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding expense: {str(e)}")

# Get all Transaction
@router.get("/expenses/")
async def get_all_expenses():
    try:
        expenses = list(expenses_collection.find().sort("date", -1))  # Sorted by latest
        for exp in expenses:
            exp["_id"] = str(exp["_id"])
        return expenses
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching all expenses: {str(e)}")
    
# 📌 Get Last Transaction
@router.get("/expenses/last")
async def get_last_expense():
    try:
        last_expense = expenses_collection.find_one(sort=[("_id", -1)])
        if not last_expense:
            raise HTTPException(status_code=404, detail="No expenses found")

        last_expense["_id"] = str(last_expense["_id"])
        return last_expense

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching last expense: {str(e)}")

# 📌 Get Today's Expenses
@router.get("/expenses/today")
async def get_today_expenses():
    try:
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)  # Next day at 00:00

        expenses = list(expenses_collection.find({
            "date": {"$gte": today_start.isoformat(), "$lt": today_end.isoformat()}
        }))

        for exp in expenses:
            exp["_id"] = str(exp["_id"])  # Convert ObjectId to string

        return expenses

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching today's expenses: {str(e)}")

# 📌 Get Monthly Expense Summary & Trend
@router.get("/expenses/monthly")
async def get_monthly_expenses():
    try:
        month_start = datetime(datetime.now().year, datetime.now().month, 1)
        next_month = datetime(datetime.now().year, datetime.now().month + 1, 1)

        transactions = list(expenses_collection.find({"date": {"$gte": month_start.isoformat(), "$lt": next_month.isoformat()}}))
        df = pd.DataFrame(transactions)

        if not df.empty:
            expenses_total = df[df["type"] == "Expense"]["amount"].sum()
            income_total = df[df["type"] == "Income"]["amount"].sum()
            expense_trend = df.groupby("date")["amount"].sum().reset_index().to_dict(orient="records")
        else:
            expenses_total = 0
            income_total = 0
            expense_trend = []

        monthly_balance = income_total - expenses_total

        return {
            "monthly_income": income_total,
            "monthly_expense": expenses_total,
            "monthly_balance": monthly_balance,
            "trend": expense_trend
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching monthly expenses: {str(e)}")
    
    
# 📌 Get Monthly Budget
@router.get("/budget/")
def get_budget():
    """Fetches the current monthly budget."""
    budget_data = budgets_collection.find_one({}, {"_id": 0, "monthly_budget": 1})
    if not budget_data:
        return {"monthly_budget": 0}
    return budget_data


# 📌 Update Monthly Budget
@router.post("/budget/")
def update_budget(data: BudgetUpdate):
    """Updates the monthly budget."""
    budgets_collection.update_one({}, {"$set": {"monthly_budget": data.amount}}, upsert=True)
    return {"message": "✅ Budget updated successfully!"}


#  📌 Get Monthly Expense Analysis
@router.get("/budget/analyze")
def get_budget_analysis():
    """Fetches and analyzes the budget vs. spending."""
    return analyze_budget()


# 📌 Get AI-Generated Budget Advice
@router.get("/budget/advice")
def get_budget_advice():
    """Generates AI-powered budget recommendations."""
    return {"advice": generate_budget_advice()}