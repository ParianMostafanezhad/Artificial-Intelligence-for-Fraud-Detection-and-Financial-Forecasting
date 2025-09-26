
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, mean_squared_error
import joblib

from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow.keras import layers, Sequential, callbacks

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")




def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)




def load_or_create_fraud_data(path=None, n_samples=20000):
    """Load fraud dataset from CSV if path is provided, otherwise generate synthetic data.
    Returns: DataFrame with feature columns and target column (1 = fraud, 0 = legit).
    """
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded data from {path}, shape={df.shape}")
        return df

    np.random.seed(42)
    n = n_samples
    X = np.random.randn(n, 8)
    fraud_prob = (0.02 + 0.18 * (np.tanh(X[:, 0] * 1.5 + X[:, 1]) > 0)).clip(0, 1)
    y = np.random.binomial(1, fraud_prob)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df['target'] = y
    print(f"Created synthetic fraud dataset, shape={df.shape}, fraud_ratio={df['target'].mean():.4f}")
    return df


def preprocess_features(df, target_col='target', scaler=None):
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    if scaler is None:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = scaler.transform(X)
    return Xs, y, scaler


def train_fraud_models(df, save_dir='models/fraud', test_size=0.2, random_state=42):
    os.makedirs(save_dir, exist_ok=True)
    X, y, scaler = preprocess_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    print("Original train fraud ratio:", y_train.mean())

    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print("After SMOTE train fraud ratio:", y_res.mean(), "Resampled shape:", X_res.shape)

    model_xgb = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    model_xgb.fit(X_res, y_res)
    preds_xgb_proba = model_xgb.predict_proba(X_test)[:, 1]
    auc_xgb = roc_auc_score(y_test, preds_xgb_proba)
    print(f"XGBoost ROC-AUC: {auc_xgb:.4f}")

    model_lgb = lgb.LGBMClassifier(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=random_state)
    model_lgb.fit(X_res, y_res)
    preds_lgb_proba = model_lgb.predict_proba(X_test)[:, 1]
    auc_lgb = roc_auc_score(y_test, preds_lgb_proba)
    print(f"LightGBM ROC-AUC: {auc_lgb:.4f}")

    input_dim = X_res.shape[1]
    model_nn = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_nn.fit(X_res, y_res, validation_split=0.1, epochs=50, batch_size=256, callbacks=[es], verbose=0)
    preds_nn_proba = model_nn.predict(X_test).ravel()
    auc_nn = roc_auc_score(y_test, preds_nn_proba)
    print(f"Neural Net ROC-AUC: {auc_nn:.4f}")

    joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
    joblib.dump(model_xgb, os.path.join(save_dir, 'xgb.joblib'))
    joblib.dump(model_lgb, os.path.join(save_dir, 'lgb.joblib'))
    model_nn.save(os.path.join(save_dir, 'nn_keras'))

    print("\nClassification report (LightGBM with 0.5 threshold):")
    y_pred_lab = (preds_lgb_proba > 0.5).astype(int)
    print(classification_report(y_test, y_pred_lab))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_lab))

    try:
        from sklearn.metrics import roc_curve
        fpr_x, tpr_x, _ = roc_curve(y_test, preds_xgb_proba)
        fpr_l, tpr_l, _ = roc_curve(y_test, preds_lgb_proba)
        fpr_n, tpr_n, _ = roc_curve(y_test, preds_nn_proba)
        plt.figure(figsize=(8,6))
        plt.plot(fpr_x, tpr_x, label=f'XGB AUC={auc_xgb:.3f}')
        plt.plot(fpr_l, tpr_l, label=f'LGB AUC={auc_lgb:.3f}')
        plt.plot(fpr_n, tpr_n, label=f'NN AUC={auc_nn:.3f}')
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_comparison.png'))
        print('Saved ROC plot to', os.path.join(save_dir, 'roc_comparison.png'))
    except Exception as e:
        print('Unable to plot ROC:', e)

    return {
        'xgb_auc': auc_xgb,
        'lgb_auc': auc_lgb,
        'nn_auc': auc_nn,
        'models_saved_to': save_dir
    }




def load_or_create_time_series(path=None, date_col='date', value_col='value', n_days=1000):
    """Load time series data if path provided, otherwise generate synthetic financial series.
    Returns: DataFrame with columns date (Datetime) and value (float).
    """
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        return df[[date_col, value_col]].rename(columns={date_col: 'date', value_col: 'value'})

    rng = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='D')
    trend = np.linspace(50, 150, n_days)
    seasonal = 10 * np.sin(np.arange(n_days) * (2 * np.pi / 30))
    noise = np.random.randn(n_days) * 3
    values = trend + seasonal + noise
    df = pd.DataFrame({'date': rng, 'value': values})
    print(f"Created synthetic time-series, last date = {df['date'].iloc[-1].date()}")
    return df


def make_supervised(series, window_size=30):
    """Convert time-series into supervised samples for neural networks.
    X: windows of length window_size
    y: next value
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)


def train_lstm_forecast(df, window_size=30, epochs=30, batch_size=32, save_dir='models/forecast', random_state=42):
    os.makedirs(save_dir, exist_ok=True)
    values = df['value'].values.astype('float32')

    mean = values.mean()
    std = values.std()
    values_norm = (values - mean) / std

    X, y = make_supervised(values_norm, window_size=window_size)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    seed_everything(random_state)

    model = Sequential([
        layers.Input(shape=(window_size, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)

    preds = model.predict(X_test).ravel()
    preds_inv = preds * std + mean
    y_test_inv = y_test * std + mean
    rmse = mean_squared_error(y_test_inv, preds_inv, squared=False)
    print(f"LSTM RMSE on test: {rmse:.4f}")

    model.save(os.path.join(save_dir, 'lstm_model'))
    joblib.dump({'mean': mean, 'std': std, 'window_size': window_size}, os.path.join(save_dir, 'scaler_params.joblib'))

    plt.figure(figsize=(10,5))
    plt.plot(df['date'][-len(y_test_inv):], y_test_inv, label='actual')
    plt.plot(df['date'][-len(preds_inv):], preds_inv, label='predicted')
    plt.legend()
    plt.title('LSTM: actual vs predicted (test tail)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lstm_pred.png'))
    print('Saved LSTM prediction plot to', os.path.join(save_dir, 'lstm_pred.png'))

    return {
        'rmse': rmse,
        'model_saved_to': save_dir,
        'history': history.history
    }


def forecast_next_days(model_dir='models/forecast', last_series=None, n_days=30):
    """Predict the next n_days using trained LSTM model and last values of the series.
    last_series must be an array of raw values (unnormalized) at least window_size long.
    """
    params = joblib.load(os.path.join(model_dir, 'scaler_params.joblib'))
    mean = params['mean']; std = params['std']; window_size = params['window_size']
    model = tf.keras.models.load_model(os.path.join(model_dir, 'lstm_model'))

    series = np.array(last_series).astype('float32')
    preds = []
    buffer = series[-window_size:].copy()
    for _ in range(n_days):
        buf_norm = (buffer - mean) / std
        x = buf_norm.reshape(1, window_size, 1)
        p = model.predict(x).ravel()[0]
        p_inv = p * std + mean
        preds.append(p_inv)
        buffer = np.roll(buffer, -1)
        buffer[-1] = p_inv
    return np.array(preds)




def main_demo():
    seed_everything(42)

    df_fraud = load_or_create_fraud_data(path=None, n_samples=15000)
    fraud_results = train_fraud_models(df_fraud, save_dir='models/fraud_example')
    print('Fraud training results:', fraud_results)

    df_ts = load_or_create_time_series(path=None, n_days=800)
    forecast_results = train_lstm_forecast(df_ts, window_size=30, epochs=20, batch_size=64, save_dir='models/forecast_example')
    print('Forecast training results:', forecast_results)

    last_vals = df_ts['value'].values
    preds30 = forecast_next_days(model_dir='models/forecast_example', last_series=last_vals, n_days=30)
    print('Next 30 days (sample):', preds30[:5])


if __name__ == '__main__':
    main_demo()