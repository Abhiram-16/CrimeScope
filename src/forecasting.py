# src/forecasting.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

# --- Define constants for file paths ---
# We are now inside the 'src' folder, so paths are relative to the project root
CSV_PATH = "data/crime_data.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "forecaster.joblib")
LAST_DATE_PATH = os.path.join(MODEL_DIR, "last_date.joblib")
LAST_INDEX_PATH = os.path.join(MODEL_DIR, "last_index.joblib")


def train_and_save_model():
    """
    This function encapsulates the entire training process from the notebook.
    It loads data, processes it, trains the model, and saves it.
    """
    print("--- Starting model training ---")
    
    # 1. Load and prepare data
    df = pd.read_csv(CSV_PATH)
    df['Incident Datetime'] = pd.to_datetime(df['Incident Datetime'], errors='coerce')
    df.dropna(subset=['Incident Datetime'], inplace=True)

    # 2. Create daily time series and filter
    df.set_index('Incident Datetime', inplace=True)
    daily_counts = df.resample('D').size()
    daily_counts_filtered = daily_counts[daily_counts.index >= '2005-01-01']

    # 3. Feature Engineering
    df_model = pd.DataFrame({'incidents': daily_counts_filtered})
    df_model['time_index'] = np.arange(len(df_model))

    y = df_model['incidents']
    X = df_model[['time_index']]

    # 4. Train the model
    model = LinearRegression()
    model.fit(X, y)
    print("Model trained successfully.")

    # 5. Save the model and helper data
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(df_model.index.max(), LAST_DATE_PATH)
    joblib.dump(X['time_index'].max(), LAST_INDEX_PATH)

    print(f"✅ Model and helper data saved to the '{MODEL_DIR}' directory.")


def predict_future(days_ahead: int) -> tuple:
    """
    Loads the saved model and predicts a future value.
    This is the function our chatbot will call.

    Args:
        days_ahead (int): Number of days into the future to predict.

    Returns:
        tuple: A tuple containing the last known date and the predicted value.
    """
    try:
        model = joblib.load(MODEL_PATH)
        last_date = joblib.load(LAST_DATE_PATH)
        last_index = joblib.load(LAST_INDEX_PATH)
    except FileNotFoundError:
        print("Model not found. Please train the model first by running this script directly.")
        return None, None

    future_index = last_index + days_ahead
    prediction = model.predict([[future_index]])

    return last_date.date(), prediction[0]


# This block allows us to run the training function directly from the command line
if __name__ == '__main__':
    # This will only run when you execute `python src/forecasting.py`
    train_and_save_model()
    
    # Example of how to use the prediction function
    print("\n--- Testing prediction function ---")
    last_known_date, future_pred = predict_future(days_ahead=30)
    if last_known_date:
        print(f"Model trained on data up to: {last_known_date}")
        print(f"Prediction for 30 days in the future: {future_pred:.2f} incidents.")