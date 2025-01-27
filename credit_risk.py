import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# 1. Generate Dummy Data
def generate_dummy_data(n_samples=1000):
    np.random.seed(42)
    data = {
        "customer_id": range(1, n_samples + 1),
        "current_balance": np.random.uniform(100, 10000, n_samples),
        "credit_limit": np.random.uniform(5000, 50000, n_samples),
        "transaction_volume": np.random.uniform(1, 100, n_samples),
        "payment_rate": np.random.uniform(0.1, 1.0, n_samples),
        "interest_rate": np.random.uniform(5, 25, n_samples),
        "age": np.random.randint(18, 80, n_samples),
        "monthly_income": np.random.uniform(2000, 20000, n_samples),
        "balance_next_month": np.random.uniform(100, 12000, n_samples)
    }
    return pd.DataFrame(data)

# 2. Load and Prepare Data
def prepare_data(data):
    features = data.drop(columns=["customer_id", "balance_next_month"])
    target = data["balance_next_month"]
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# 3. Train the Model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 4. Evaluate the Model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, r2, predictions

# 5. Save Data and Model
def save_data_and_model(data, X_train, X_test, y_train, y_test, predictions, model, mse, r2):
    # Save data to CSV files
    data.to_csv("dummy_data.csv", index=False)
    pd.DataFrame(X_train).to_csv("train_features.csv", index=False)
    pd.DataFrame(X_test).to_csv("test_features.csv", index=False)
    pd.DataFrame({"Actual": y_test.values, "Predicted": predictions}).to_csv("predictions.csv", index=False)
    
    # Save model and metrics
    joblib.dump(model, "balance_forecasting_model.pkl")
    metrics = {"mse": mse, "r2_score": r2}
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f)

# 6. Visualization
def visualize_predictions(y_test, predictions):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.6, color='blue', label='Predictions vs. Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
    plt.title("Predicted vs. Actual Balances")
    plt.xlabel("Actual Balance Next Month")
    plt.ylabel("Predicted Balance Next Month")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("predictions_vs_actuals.png")
    plt.show()

# Main Pipeline
if __name__ == "__main__":
    # Step 1: Generate Dummy Data
    data = generate_dummy_data()
    print(f"Sample Data:\n{data.head()}")

    # Step 2: Prepare Data
    X_train, X_test, y_train, y_test = prepare_data(data)

    # Step 3: Train the Model
    model = train_model(X_train, y_train)
    print("Model training completed.")

    # Step 4: Evaluate the Model
    mse, r2, predictions = evaluate_model(model, X_test, y_test)
    print(f"Model Evaluation:\nMean Squared Error: {mse}\nR2 Score: {r2}")

    # Step 5: Save Data and Model
    save_data_and_model(data, X_train, X_test, y_train, y_test, predictions, model, mse, r2)
    print("Data, model, and metrics saved successfully.")

    # Step 6: Visualize Predictions
    visualize_predictions(y_test, predictions)
    print("Visualization saved as predictions_vs_actuals.png.")
