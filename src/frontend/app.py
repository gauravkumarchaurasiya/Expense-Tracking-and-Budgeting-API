import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.title("💰 Expense Tracker")

# 📌 Get Budget
budget_res = requests.get(f"{API_URL}/budget/")
budget = budget_res.json().get("monthly_budget", 0)
st.sidebar.metric("📊 Monthly Budget", f"₹ {budget}")

# 📌 Enter Expense
st.header("📌 Add Expense")
title = st.text_input("Title")
amount = st.number_input("Amount", min_value=0.01)
account = st.selectbox("Account", ["Cash", "Bank", "Credit Card"])
type_ = st.selectbox("Type", ["Expense", "Income"])

if st.button("Add Expense"):
    data = {"title": title, "amount": amount, "account": account, "type": type_}
    res = requests.post(f"{API_URL}/expenses/", json=data)

    if res.status_code == 200:
        st.success("✅ Expense added successfully!")
    else:
        st.error("⚠️ Failed to add expense!")

# 📌 Display Last Transaction
st.header("📌 Last Transaction")
last_expense = requests.get(f"{API_URL}/expenses/last").json()
if last_expense:
    st.write(f"**{last_expense['title']}** - ₹{last_expense['amount']} ({last_expense['type']}) | Category: **{last_expense['category']}**")

# 📌 Today's Expenses
st.header("📅 Today's Expenses")
today_expenses = requests.get(f"{API_URL}/expenses/today").json()
if today_expenses:
    df_today = pd.DataFrame(today_expenses)
    st.dataframe(df_today[["title", "amount", "account", "type", "category"]])

# 📌 Monthly Expenses Chart
st.header("📊 Monthly Expense Trend")
monthly_expenses = requests.get(f"{API_URL}/expenses/monthly").json()
monthly_expense_total = monthly_expenses.get("monthly_expense", 0)

if monthly_expense_total > budget:
    st.warning("⚠️ You have exceeded your monthly budget!")
else:
    st.success(f"✅ Remaining Budget: ₹{budget - monthly_expense_total}")

# 📌 Balance Calculation
balance = budget - monthly_expense_total
st.sidebar.metric("💰 Balance Left", f"₹ {balance}")

# 📌 Set Monthly Budget
st.sidebar.header("Set Monthly Budget")
new_budget = st.sidebar.number_input("New Budget", min_value=0)
if st.sidebar.button("Update Budget"):
    res = requests.post(f"{API_URL}/budget/", json={"amount": new_budget})
    if res.status_code == 200:
        st.sidebar.success("✅ Budget updated!")
