import os
import pandas as pd
import glob
import sqlite3
from sklearn.model_selection import train_test_split
import pickle


def load_and_merge_data(data_path='data'):
    all_files = glob.glob(os.path.join(data_path, '*.pkl'))
    df_list = [pd.read_pickle(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    return df


def add_time_features(df):
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
    df['TX_DAY'] = df['TX_DATETIME'].dt.day
    df['TX_WEEKDAY'] = df['TX_DATETIME'].dt.weekday
    return df


def compute_terminal_fraud_rolling(df, window=7):
    df = df.sort_values('TX_DATETIME')
    df['TERMINAL_FRAUD_7D'] = (
        df.groupby('TERMINAL_ID')['TX_FRAUD']
        .rolling(window=window, min_periods=1)
        .sum()
        .shift(1)
        .fillna(0)
        .reset_index(level=0, drop=True)
    )
    return df


def compute_customer_spend_rolling(df, window=7):
    df = df.sort_values('TX_DATETIME')
    df['CUSTOMER_SPEND_7D'] = (
        df.groupby('CUSTOMER_ID')['TX_AMOUNT']
        .rolling(window=window, min_periods=1)
        .sum()
        .shift(1)
        .fillna(0)
        .reset_index(level=0, drop=True)
    )
    return df


def save_to_database(df, db_path='processed/transactions.db'):
    os.makedirs('processed', exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql('transactions', conn, if_exists='replace', index=False)


def preprocess_data():
    df = load_and_merge_data('data')
    df = add_time_features(df)
    df = compute_terminal_fraud_rolling(df)
    df = compute_customer_spend_rolling(df)

    features = [
        'TX_AMOUNT', 'TX_HOUR', 'TX_DAY', 'TX_WEEKDAY',
        'TERMINAL_FRAUD_7D', 'CUSTOMER_SPEND_7D',
        'CUSTOMER_ID', 'TERMINAL_ID', 'TX_DATETIME', 'TX_FRAUD'
    ]
    df = df[features]
    df = df.sort_values('TX_DATETIME')

    save_to_database(df)

    # Train/test split for training the model
    X = df.drop(columns='TX_FRAUD')
    y = df['TX_FRAUD']
    split_idx = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    with open('processed/X_train.pkl', 'wb') as f: pickle.dump(X_train, f)
    with open('processed/X_test.pkl', 'wb') as f: pickle.dump(X_test, f)
    with open('processed/y_train.pkl', 'wb') as f: pickle.dump(y_train, f)
    with open('processed/y_test.pkl', 'wb') as f: pickle.dump(y_test, f)

    print("✅ Preprocessing complete. Data saved to SQLite and 'processed/' folder.")


if __name__ == '__main__':
    preprocess_data()
