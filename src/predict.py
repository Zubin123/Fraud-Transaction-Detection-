import os
import pandas as pd
import joblib
import pickle
from preprocessing import add_time_features, compute_terminal_fraud_rolling, compute_customer_spend_rolling

def load_model(path='models/fraud_model.pkl'):
    model = joblib.load(path)
    return model

def load_new_data(file_path):
    return pd.read_pickle(file_path)

def preprocess_new_data(df):
    df = add_time_features(df)
    df = compute_terminal_fraud_rolling(df)
    df = compute_customer_spend_rolling(df)

    features = [
        'TX_AMOUNT', 'TX_HOUR', 'TX_DAY', 'TX_WEEKDAY',
        'TERMINAL_FRAUD_7D', 'CUSTOMER_SPEND_7D'
    ]
    return df[features]

def predict_fraud(model, processed_df):
    predictions = model.predict(processed_df)
    probabilities = model.predict_proba(processed_df)[:, 1]
    return predictions, probabilities

def main():
    model = load_model()
    df = load_new_data('data/2018-09-25.pkl')
    original_df = df.copy()

    processed_df = preprocess_new_data(df)
    preds, probs = predict_fraud(model, processed_df)

    original_df['PREDICTED_FRAUD'] = preds
    original_df['FRAUD_PROBABILITY'] = probs

    print(original_df[['TRANSACTION_ID', 'TX_AMOUNT', 'PREDICTED_FRAUD', 'FRAUD_PROBABILITY']].head())

    # ✅ Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    original_df.to_csv('results/predictions.csv', index=False)
    print("\n✅ Predictions saved to 'results/predictions.csv'")


if __name__ == '__main__':
    main()
