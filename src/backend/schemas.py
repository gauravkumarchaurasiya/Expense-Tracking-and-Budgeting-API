from pydantic import BaseModel
from datetime import datetime

class ExpenseCreate(BaseModel):
    title: str
    amount: float
    account: str
    type: str

class Expense(ExpenseCreate):
    date: datetime
    category: str
    
class BudgetInput(BaseModel):
    amount: float
    
class BudgetUpdate(BaseModel):
    amount: float

# class PredictionInput(BaseModel):
#     title: str
#     amount: float
#     account_encoded: int
#     type_encoded: int