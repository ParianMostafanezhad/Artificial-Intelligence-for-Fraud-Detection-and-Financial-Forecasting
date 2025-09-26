Fraud Detection and Financial Forecasting with AI
 Overview

This project implements Fraud Detection and Financial Time-Series Forecasting using Artificial Intelligence.
It combines traditional machine learning models (XGBoost, LightGBM, Random Forest) with deep learning (Neural Networks, LSTM) to provide a robust framework for academic research and real-world financial applications.

 Features

Fraud Detection

Synthetic or CSV-based transaction data

Data preprocessing and scaling

Handling class imbalance with SMOTE

Models: XGBoost, LightGBM, Neural Network (Keras)

Metrics: ROC-AUC, classification report, confusion matrix

ROC comparison plots and saved models

Financial Forecasting

Synthetic or CSV-based time-series data

Preprocessing with sliding window approach

LSTM-based deep learning model for trend/seasonality forecasting

Evaluation with RMSE

Prediction of future values (n-days ahead)

Visualization of actual vs predicted values

 Tech Stack

Python 3.9+

Libraries:

Machine Learning: scikit-learn, xgboost, lightgbm, imblearn

Deep Learning: tensorflow / keras

Utilities: numpy, pandas, joblib, matplotlib, seaborn

📂 Project Structure
├── models/                # Saved models and plots  
│   ├── fraud_example/  
│   └── forecast_example/  
├── fraud_forecast_ai.py   # Main source code  
├── README.md              # Project documentation  

 Usage

Clone the repository:

git clone https://github.com/your-username/fraud-forecast-ai.git
cd fraud-forecast-ai


Install dependencies:

pip install -r requirements.txt


Run demo (fraud detection + forecasting):

python fraud_forecast_ai.py


Example prediction (30 days ahead):

from fraud_forecast_ai import forecast_next_days, load_or_create_time_series
df = load_or_create_time_series(n_days=800)
preds = forecast_next_days(model_dir="models/forecast_example", last_series=df["value"].values, n_days=30)
print(preds)

 Results

Fraud Detection ROC-AUC (synthetic data): ~0.90+

LSTM Forecasting RMSE: varies depending on dataset

Plots are saved under models/.../ for visualization
