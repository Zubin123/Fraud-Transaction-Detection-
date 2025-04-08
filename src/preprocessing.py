import os
import pandas as pd
import glob
import pickle

from sklearn.model_selection import train_test_split


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


def compute_terminal_fraud_rolling(df, window='7d'):
    df = df.sort_values('TX_DATETIME')
    results = []
    for terminal_id, group in df.groupby('TERMINAL_ID'):
        group = group.set_index('TX_DATETIME').sort_index()
        group['TERMINAL_FRAUD_7D'] = group['TX_FRAUD'].rolling(window).sum().shift(1).fillna(0)
        results.append(group.reset_index())
    return pd.concat(results, ignore_index=True)


def compute_customer_spend_rolling(df, window='7d'):
    df = df.sort_values('TX_DATETIME')
    results = []
    for customer_id, group in df.groupby('CUSTOMER_ID'):
        group = group.set_index('TX_DATETIME').sort_index()
        group['CUSTOMER_SPEND_7D'] = group['TX_AMOUNT'].rolling(window).mean().shift(1).fillna(0)
        results.append(group.reset_index())
    return pd.concat(results, ignore_index=True)


def preprocess_data():
    df = load_and_merge_data('data')
    df = add_time_features(df)
    df = compute_terminal_fraud_rolling(df)
    df = compute_customer_spend_rolling(df)

    # Final features
    features = [
        'TX_AMOUNT', 'TX_HOUR', 'TX_DAY', 'TX_WEEKDAY',
        'TERMINAL_FRAUD_7D', 'CUSTOMER_SPEND_7D'
    ]
    target = 'TX_FRAUD'

    df = df.sort_values('TX_DATETIME')
    X = df[features]
    y = df[target]

    split_idx = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    os.makedirs('processed', exist_ok=True)
    with open('processed/X_train.pkl', 'wb') as f: pickle.dump(X_train, f)
    with open('processed/X_test.pkl', 'wb') as f: pickle.dump(X_test, f)
    with open('processed/y_train.pkl', 'wb') as f: pickle.dump(y_train, f)
    with open('processed/y_test.pkl', 'wb') as f: pickle.dump(y_test, f)

    print("✅ Preprocessing complete. Data saved to 'processed/' folder.")


if __name__ == '__main__':
    preprocess_data()
