import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import datetime, time

# Load trained model
model = joblib.load("models/fraud_model.pkl")

# Load full preprocessed dataset into SQLite (or connect if already exists)
conn = sqlite3.connect("processed/transactions.db")

# Create table if it doesn't exist
conn.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    TX_DATETIME TEXT,
    TX_AMOUNT REAL,
    CUSTOMER_ID TEXT,
    TERMINAL_ID TEXT,
    TX_HOUR INTEGER,
    TX_DAY INTEGER,
    TX_WEEKDAY INTEGER,
    CUSTOMER_SPEND_7D REAL,
    TERMINAL_FRAUD_7D REAL,
    PREDICTED_LABEL INTEGER,
    FRAUD_PROBABILITY REAL
)
""")
conn.commit()


st.set_page_config(page_title="Fraud Transaction Detection", layout="centered")
st.title("💳 Fraud Transaction Detection")

st.header("📥 Transaction Input")
amount = st.number_input("Transaction Amount", min_value=0.01, value=100.00)
customer_id = st.text_input("Customer ID", "2403")
terminal_id = st.text_input("Terminal ID", "4018")
date_input = st.date_input("Transaction Date", value=datetime(2018, 4, 2).date())
time_input = st.time_input("Transaction Time", value=time(12, 0))

if st.button("Detect Fraud"):
    tx_datetime = datetime.combine(date_input, time_input)

    # Create transaction record
    new_tx = pd.DataFrame([{
        'TX_DATETIME': tx_datetime,
        'TX_AMOUNT': amount,
        'CUSTOMER_ID': customer_id,
        'TERMINAL_ID': terminal_id
    }])

    # Add time features
    new_tx['TX_HOUR'] = tx_datetime.hour
    new_tx['TX_DAY'] = tx_datetime.day
    new_tx['TX_WEEKDAY'] = tx_datetime.weekday()

    # Compute rolling features via SQL
    query = f"""
        SELECT SUM(TX_AMOUNT) AS CUSTOMER_SPEND_7D
        FROM transactions
        WHERE CUSTOMER_ID = '{customer_id}'
        AND TX_DATETIME < '{tx_datetime}'
        AND TX_DATETIME >= datetime('{tx_datetime}', '-7 days')
    """
    spend_7d = conn.execute(query).fetchone()[0] or 0
    new_tx['CUSTOMER_SPEND_7D'] = spend_7d

    query = f"""
        SELECT SUM(TX_FRAUD) AS TERMINAL_FRAUD_7D
        FROM transactions
        WHERE TERMINAL_ID = '{terminal_id}'
        AND TX_DATETIME < '{tx_datetime}'
        AND TX_DATETIME >= datetime('{tx_datetime}', '-7 days')
    """
    fraud_7d = conn.execute(query).fetchone()[0] or 0
    new_tx['TERMINAL_FRAUD_7D'] = fraud_7d

    # Feature selection
    features = ['TX_AMOUNT', 'TX_HOUR', 'TX_DAY', 'TX_WEEKDAY',
                'TERMINAL_FRAUD_7D', 'CUSTOMER_SPEND_7D']
    X = new_tx[features]

    import matplotlib.pyplot as plt

    # --- Visualization of Customer Spend ---
    query_spend = f"""
        SELECT TX_DATETIME, TX_AMOUNT
        FROM transactions
        WHERE CUSTOMER_ID = '{customer_id}'
        AND TX_DATETIME < '{tx_datetime}'
        AND TX_DATETIME >= datetime('{tx_datetime}', '-7 days')
        ORDER BY TX_DATETIME
    """
    df_spend = pd.read_sql(query_spend, conn, parse_dates=["TX_DATETIME"])

    if not df_spend.empty:
        st.subheader("📊 Past 7 Days Customer Spending")
        fig, ax = plt.subplots()
        ax.plot(df_spend['TX_DATETIME'], df_spend['TX_AMOUNT'], marker='o')
        ax.set_ylabel("Transaction Amount")
        ax.set_xlabel("Date")
        ax.set_title(f"Customer {customer_id} Spend Trend")
        st.pyplot(fig)

    # --- Visualization of Terminal Fraud ---
    query_fraud = f"""
        SELECT TX_DATETIME, TX_FRAUD
        FROM transactions
        WHERE TERMINAL_ID = '{terminal_id}'
        AND TX_DATETIME < '{tx_datetime}'
        AND TX_DATETIME >= datetime('{tx_datetime}', '-7 days')
        ORDER BY TX_DATETIME
    """
    df_fraud = pd.read_sql(query_fraud, conn, parse_dates=["TX_DATETIME"])

    if not df_fraud.empty and df_fraud['TX_FRAUD'].sum() > 0:
        st.subheader("⚠️ Terminal Fraud History (Last 7 Days)")
        df_fraud = df_fraud[df_fraud['TX_FRAUD'] == 1]
        fig2, ax2 = plt.subplots()
        ax2.hist(df_fraud['TX_DATETIME'], bins=7, edgecolor='black')
        ax2.set_ylabel("Fraud Count")
        ax2.set_xlabel("Date")
        ax2.set_title(f"Fraudulent Tx on Terminal {terminal_id}")
        st.pyplot(fig2)

    # Prediction
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    # Save input + prediction to database
    conn.execute("""
    INSERT INTO predictions (
        TX_DATETIME, TX_AMOUNT, CUSTOMER_ID, TERMINAL_ID,
        TX_HOUR, TX_DAY, TX_WEEKDAY,
        CUSTOMER_SPEND_7D, TERMINAL_FRAUD_7D,
        PREDICTED_LABEL, FRAUD_PROBABILITY
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
    tx_datetime, amount, customer_id, terminal_id,
    new_tx['TX_HOUR'].values[0], new_tx['TX_DAY'].values[0], new_tx['TX_WEEKDAY'].values[0],
    new_tx['CUSTOMER_SPEND_7D'].values[0], new_tx['TERMINAL_FRAUD_7D'].values[0],
    int(pred), float(prob)
    ))
    conn.commit()


    # Display
    st.subheader("📊 Transaction Details:")
    st.write(f"**TX_DATETIME:** {tx_datetime}")
    st.write(f"**TX_AMOUNT:** {amount}")
    st.write(f"**CUSTOMER_ID:** {customer_id}")
    st.write(f"**TERMINAL_ID:** {terminal_id}")

    st.subheader("🧠 Computed Features for Prediction:")
    for f in features[1:]:  # skip TX_AMOUNT since it's already above
        st.write(f"**{f.upper()}:** {new_tx[f].values[0]}")

    if pred == 1:
        st.error(f"⚠️ Transaction is FRAUD (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Transaction is Legit (Probability of fraud: {prob:.2f})")

with st.expander("📋 View Saved Predictions"):
    df_saved = pd.read_sql("SELECT * FROM predictions ORDER BY TX_DATETIME DESC LIMIT 50", conn)
    st.dataframe(df_saved)
