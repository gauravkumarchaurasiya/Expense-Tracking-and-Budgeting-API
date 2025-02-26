import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

st.title("💰 AI-Powered Expense Tracker & Budget Advisor")

# 📌 Fetch Monthly Budget
budget_res = requests.get(f"{API_URL}/budget/")
budget = budget_res.json().get("monthly_budget", 0)
st.sidebar.metric("📊 Monthly Budget", f"₹ {budget}")

# 📌 Enter Expense
st.header("📌 Add   Expense")
title = st.text_input("Title")
amount = st.number_input("Amount", min_value=0.01)
# category = st.selectbox("Category", [
#     "Food & Dining", "Transport", "Shopping", "Utilities",
#     "Medical & Healthcare", "Entertainment & Leisure",
#     "Rent & Housing", "Miscellaneous"
# ])
account = st.selectbox("Account", ["Cash", "Online"])
type_ = st.selectbox("Type", ["Expense", "Income"])

if st.button("Add Expense"):
    data = {"title": title, "amount": amount,  "account": account, "type": type_}
    res = requests.post(f"{API_URL}/expenses/", json=data)

    if res.status_code == 200:
        st.success("✅ Expense added successfully!")
    else:
        st.error("⚠️ Failed to add expense!")
        
# 📌 Last Transaction
st.header("📌 Last Transaction")
last_expense = requests.get(f"{API_URL}/expenses/last").json()
if last_expense:
    st.write(f"**{last_expense['title']}** - ₹{last_expense['amount']} ({last_expense['type']}) | Category: **{last_expense['category']}**")
    
# 📌 Display All Transactions in Scrollable Format
st.header("📜 All Transactions")
all_transactions = requests.get(f"{API_URL}/expenses/").json()

if isinstance(all_transactions, dict):  
    # If it's a single dictionary, wrap it in a list
    all_transactions = [all_transactions]  
elif not isinstance(all_transactions, list):
    all_transactions = []  # Ensure an empty list if response is invalid

# Convert to DataFrame only if there are transactions
if all_transactions:
    df_transactions = pd.DataFrame(all_transactions)

    # Ensure the required columns exist before displaying
    expected_columns = ["date", "title", "amount", "account", "type", "category"]
    df_transactions = df_transactions[[col for col in expected_columns if col in df_transactions.columns]]

    st.dataframe(df_transactions, height=300)  # Scrollable
else:
    st.info("No transactions available.")


# 📌 Today's Expenses
st.header("📅 Today's Expenses")
today_expenses = requests.get(f"{API_URL}/expenses/today").json()

if today_expenses:
    df_today = pd.DataFrame(today_expenses)
    df_today["category"] = df_today["category"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))

    st.dataframe(df_today[["title", "amount", "account", "type", "category"]])

    # 🎨 Pie Chart for Today's Expenses
    st.subheader("📊 Today's Expense Distribution")
    pie_today = px.pie(df_today, values="amount", names="category", title="Today's Expenses by Category")
    st.plotly_chart(pie_today)
else:
    st.info("No expenses recorded today.")

# 📌 Monthly Expenses Trend
st.header("📊 Monthly Expense Trend")
monthly_expenses = requests.get(f"{API_URL}/expenses/monthly").json()
monthly_expense_total = monthly_expenses.get("monthly_expense", 0)
expense_trend = monthly_expenses.get("trend", [])  # Fetching trend data (list of amounts over time)

# # 🎨 Pie Chart for Monthly Expenses
# st.subheader("📊 Monthly Expense Breakdown")
df_monthly = pd.DataFrame(expense_trend, columns=["date", "amount", "category"])
# if not df_monthly.empty:
#     pie_monthly = px.pie(df_monthly, values="amount", names="category", title="Monthly Expenses by Category")
#     st.plotly_chart(pie_monthly)

# 📈 Line Chart for Monthly Expense Trend
st.subheader("📈 Monthly Expense Increase/Decrease")
if not df_monthly.empty:
    line_chart = px.line(df_monthly, x="date", y="amount", title="Monthly Expense Trend", markers=True)
    st.plotly_chart(line_chart)

# 📌 Budget Warning or Success Message
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

# 📌 Budget Analysis & AI Advice
st.header("🧠 AI-Powered Budget Analysis")

# Fetch Budget Analysis Data
analysis_res = requests.get(f"{API_URL}/budget/analyze").json()

if isinstance(analysis_res, dict):
    st.write("### 🔍 Budget Breakdown")
    
    year, month, analysis_data = analysis_res["year"], analysis_res["month"], analysis_res["analysis"]
    
    st.subheader(f"📅 {month}/{year}")

    # Convert analysis to DataFrame
    df_analysis = pd.DataFrame.from_dict(analysis_data, orient="index")

    # Display total spent and remaining budget
    total_spent = df_analysis["spent"].sum()
    total_budget = df_analysis["budget"].sum()
    remaining_budget = total_budget - total_spent

    col1, col2, col3 = st.columns(3)
    col1.metric("💸 Total Spent", f"₹ {total_spent}")
    col2.metric("🏦 Budget", f"₹ {total_budget}")
    col3.metric("✅ Remaining", f"₹ {remaining_budget}")

    # Color Coding for Status
    df_analysis["status_color"] = df_analysis["status"].apply(lambda x: "🟢 Within Budget" if x == "Within Budget" else "🔴 Over Budget")

    # Prettier Dataframe Display
    st.dataframe(df_analysis[["spent", "budget", "status_color"]].rename(columns={
        "spent": "💰 Spent",
        "budget": "🏦 Budget",
        "status_color": "📊 Status"
    }))

    # Bar Chart for Visual Representation
    st.subheader("📊 Spending vs Budget")
    st.bar_chart(df_analysis[["spent", "budget"]])



    # Fetch AI financial advice
    advice_res = requests.get(f"{API_URL}/budget/advice").json()
    if advice_res:
        st.subheader("💡 AI Financial Advice")
        st.write(f"**{advice_res.get('advice', 'No advice available')}**")
else:
    st.warning("⚠️ No transaction data found for this month!")

