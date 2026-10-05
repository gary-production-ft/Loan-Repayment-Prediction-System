import joblib
import numpy as np

def main():
    print("=======================================")
    print("   LOAN REPAYMENT PREDICTION SYSTEM    ")
    print("=======================================")
    
    try:
        model = joblib.load("loan_model.pkl")
    except FileNotFoundError:
        print("Error: loan_model.pkl not found. Please run train_model.py first.")
        return
        
    print("\nPlease enter customer details:")
    try:
        monthly_income = float(input("Monthly Income (₹): "))
        credit_score = float(input("Credit Score (300-900): "))
        employment_years = float(input("Employment Years: "))
        monthly_expenses = float(input("Monthly Expenses (₹): "))
        existing_loans = int(input("Existing Loans: "))
        previous_loan_count = int(input("Previous Loan Count: "))
        previous_loan_defaults = int(input("Previous Loan Defaults: "))
        loan_amount = float(input("Loan Amount (₹): "))
        loan_term = int(input("Loan Term (months - e.g., 12, 24, 36, 48, 60): "))
        
        if previous_loan_defaults > previous_loan_count:
            print("\nError: Previous defaults cannot be greater than previous loan count.")
            return
            
        input_data = np.array([[
            monthly_income,
            credit_score,
            employment_years,
            monthly_expenses,
            existing_loans,
            previous_loan_count,
            previous_loan_defaults,
            loan_amount,
            loan_term
        ]])
        
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        prob_not_repay = probabilities[0] * 100
        prob_repay = probabilities[1] * 100
        
        print("\n-----------------------------------")
        print("        LOAN PREDICTION")
        print("-----------------------------------")
        if prediction == 1:
            print("Prediction: LIKELY TO REPAY")
        else:
            print("Prediction: MAY NOT REPAY")
            
        print(f"\nRepayment Probability: {prob_repay:.1f}%")
        print(f"Non-Repayment Probability: {prob_not_repay:.1f}%")
        print("-----------------------------------")
        
        print("\nWhy this prediction?")
        print("The prediction is mainly influenced by the customer's")
        print("credit score, income, previous defaults and loan amount.")
        
    except ValueError:
        print("Invalid input! Please enter numerical values only.")

if __name__ == "__main__":
    main()
