import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

def load_data():
    with open('processed/X_train.pkl', 'rb') as f: X_train = pickle.load(f)
    with open('processed/y_train.pkl', 'rb') as f: y_train = pickle.load(f)
    with open('processed/X_test.pkl', 'rb') as f: X_test = pickle.load(f)
    with open('processed/y_test.pkl', 'rb') as f: y_test = pickle.load(f)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',  # handles imbalance
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))
    print("\n🔢 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n📈 ROC AUC Score: {:.4f}".format(roc_auc_score(y_test, y_proba)))

def save_model(model, path='models/fraud_model.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\n💾 Model saved to {path}")

def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)

if __name__ == '__main__':
    main()
