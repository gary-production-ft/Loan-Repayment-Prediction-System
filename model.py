import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

def create_synthetic_data(filename="loan_data.csv", n_samples=3000):
    np.random.seed(42)
    
    # Generate features
    income = np.random.randint(20000, 200000, n_samples)
    credit_score = np.random.randint(300, 900, n_samples)
    loan_amount = np.random.randint(50000, 2000000, n_samples)
    loan_term = np.random.choice([12, 24, 36, 48, 60], n_samples)
    employment_years = np.random.randint(0, 40, n_samples)
    existing_loans = np.random.randint(0, 5, n_samples)
    monthly_expenses = np.random.randint(10000, 80000, n_samples)
    previous_loan_count = np.random.randint(0, 5, n_samples)
    
    # previous defaults should not exceed previous loan count
    previous_loan_defaults = np.array([np.random.randint(0, count + 1) if count > 0 else 0 for count in previous_loan_count])
    
    # Simple logic for target
    # Higher credit score, higher income, lower loan amount, fewer defaults -> more likely to repay
    score = (credit_score / 900) * 0.4 + (income / 200000) * 0.2 - (loan_amount / 2000000) * 0.2 - (previous_loan_defaults / 5) * 0.3 + (employment_years / 40) * 0.1
    
    # Add some noise
    score += np.random.normal(0, 0.1, n_samples)
    
    # Threshold for repayment
    loan_repaid = (score > 0.25).astype(int)
    
    data = pd.DataFrame({
        'monthly_income': income,
        'credit_score': credit_score,
        'employment_years': employment_years,
        'monthly_expenses': monthly_expenses,
        'existing_loans': existing_loans,
        'previous_loan_count': previous_loan_count,
        'previous_loan_defaults': previous_loan_defaults,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'loan_repaid': loan_repaid
    })
    
    # Add a few missing values to demonstrate preprocessing
    # Only for some numerical columns
    mask = np.random.rand(n_samples) < 0.05
    data.loc[mask, 'credit_score'] = np.nan
    
    data.to_csv(filename, index=False)
    print(f"Generated {n_samples} rows of synthetic data in {filename}")

def load_data(filename="loan_data.csv"):
    if not os.path.exists(filename):
        create_synthetic_data(filename)
    
    data = pd.read_csv(filename)
    
    # Basic preprocessing: fill missing values
    data['credit_score'] = data['credit_score'].fillna(data['credit_score'].median())
    
    return data

def train_model(X_train, y_train):
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }
    
    return metrics

def predict_loan(model, customer_data):
    # Ensure columns match training data order
    features = [
        customer_data['monthly_income'],
        customer_data['credit_score'],
        customer_data['employment_years'],
        customer_data['monthly_expenses'],
        customer_data['existing_loans'],
        customer_data['previous_loan_count'],
        customer_data['previous_loan_defaults'],
        customer_data['loan_amount'],
        customer_data['loan_term']
    ]
    
    # Reshape for single prediction
    features_array = np.array(features).reshape(1, -1)
    
    prediction = model.predict(features_array)[0]
    probabilities = model.predict_proba(features_array)[0]
    
    return prediction, probabilities
