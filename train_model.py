import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib
from model import load_data, train_model, evaluate_model

def main():
    print("Loading and preprocessing data...")
    data = load_data("loan_data.csv")
    
    X = data.drop("loan_repaid", axis=1)
    y = data["loan_repaid"]
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Decision Tree model...")
    model = train_model(X_train, y_train)
    
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
        
    print("Calculating Confusion Matrix...")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save the model
    print("Saving model to loan_model.pkl...")
    joblib.dump(model, "loan_model.pkl")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
