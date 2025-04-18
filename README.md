
# 💳 Fraud Transaction Detection App

A real-time fraud detection system built using **Streamlit**, **SQLite**, and **scikit-learn**. This project allows users to input new transaction data, compute dynamic rolling features from historical transaction data, and detect fraud using a trained machine learning model. All user inputs and prediction results are saved for analysis.

---

## 📌 Features

- 📥 **Live Transaction Input** via an intuitive form
- 🧠 **Rolling Feature Calculation** (7-day historical data)
  - `CUSTOMER_SPEND_7D`
  - `TERMINAL_FRAUD_7D`
- 🔍 **Real-Time Fraud Prediction** using a trained classifier
- 💾 **Input & Prediction Logging** to SQLite/CSV
- 📊 **Historical Pattern Extraction** via database queries

---

## 🛠️ Technologies Used

- Python 3.9+
- Streamlit
- SQLite (for fast, lightweight querying)
- scikit-learn
- pandas, joblib, datetime, etc.

---

## 📁 Project Structure

```
Fraud-Transaction-Detection/
├── data/                         
├── processed/                   
│   ├── transactions.db
│   ├── X_train.pkl / X_test.pkl
│   └── y_train.pkl / y_test.pkl
├── models/                      
│   └── fraud_model.pkl
├── streamlit_app.py            
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           
│   ├── train_model.py            
│   └── predict.py
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Fraud-Transaction-Detection.git
cd Fraud-Transaction-Detection
```

### 2. Set Up the Environment

Create and activate a virtual environment (optional):

```bash
conda create -n venv python=3.9
conda activate venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing

Ensure your `.pkl` data files are inside the `data/` folder, then run:

```bash
python preprocessing.py
```

This will:
- Merge and process the dataset
- Add time features
- Save to a SQLite database (`processed/transactions.db`)
- Split train/test and save for model training

### 4. Train the Model

Train a model using the preprocessed features (`CUSTOMER_SPEND_7D`, `TERMINAL_FRAUD_7D`, etc.). Save it to `models/fraud_model.pkl`.

### 5. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

---

## 🧪 Example

- **TX_AMOUNT**: 208.00  
- **CUSTOMER_ID**: 4336  
- **TERMINAL_ID**: 4485  
- **TX_DATETIME**: 2018-09-30 15:08  

**Prediction Result**:
> ⚠️ Transaction is FRAUD (Probability: 0.54)  
> Computed Rolling Features:  
> - CUSTOMER_SPEND_7D: 281.05  
> - TERMINAL_FRAUD_7D: 14

---

## 💾 Data Logging

All transactions submitted via the form (with predictions) are logged into a separate SQLite table.

---

## ✍️ Author

**Mohammed Zubin Essudeen**
