# 🎓 Loan Repayment Prediction System

## 1. Project Title
Loan Repayment Prediction System (B.Tech AIML Mini-Project)

## 2. Problem Statement
Banks and financial institutions receive numerous loan applications daily. Manually evaluating every customer's ability to repay is a time-consuming and error-prone process. This project aims to automate the initial assessment by predicting whether a customer will repay a loan based on historical data using Machine Learning.

## 3. Objective
To build a student-level Machine Learning model that predicts loan repayment likelihood (where `1 = Repaid`, `0 = Not Repaid`) based on basic customer and loan details. The project focuses on learning the complete ML pipeline: data creation, preprocessing, training, evaluation, and deployment using a Graphical User Interface (GUI).

## 4. How the AI/ML Model is Created & Trained

The project follows a standard Machine Learning pipeline implemented in Python using Scikit-Learn.

### Step-by-Step Flow:
1. **Data Generation/Loading**: We use a synthetic dataset `loan_data.csv` with 3000 records.
2. **Data Preprocessing**: We handle missing values (e.g., empty credit scores) by filling them with the median value of the column.
3. **Splitting the Data**: We separate the dataset into Features (`X`) and Target (`y`). Then, we split them into Training Data (80%) and Testing Data (20%).
4. **Model Training**: We train a `DecisionTreeClassifier` on the Training Data.
5. **Model Evaluation**: We evaluate the trained model on the Testing Data to see its Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
6. **Saving the Model**: The trained model is exported as a file (`loan_model.pkl`) so that it doesn't need to be retrained every time a user wants to make a prediction.

---

## 5. Function Descriptions

Here is a breakdown of the important functions used in the project, written in a way that is easy to explain during a viva/presentation:

### Inside `model.py` (Core ML Logic)

*   `create_synthetic_data(filename, n_samples)`:
    *   **Purpose**: Automatically generates a realistic dataset (`loan_data.csv`) if one doesn't exist.
    *   **How it works**: Uses `numpy.random` to generate random realistic values for features like income and loan amount. It calculates a logical 'score' (higher credit score = higher chance to repay) and uses it to determine the `loan_repaid` target variable.
*   `load_data(filename)`:
    *   **Purpose**: Loads the dataset and performs basic data preprocessing.
    *   **How it works**: Uses Pandas `pd.read_csv()` to load the CSV. It then uses `.fillna(median)` to handle any missing numerical data (like missing credit scores) before passing the data to the model.
*   `train_model(X_train, y_train)`:
    *   **Purpose**: Trains the Machine Learning algorithm.
    *   **How it works**: Initializes a `DecisionTreeClassifier` from Scikit-Learn with a `max_depth=5` (to keep the tree simple and prevent overfitting). It then calls `.fit(X_train, y_train)` to make the model learn the patterns.
*   `evaluate_model(model, X_test, y_test)`:
    *   **Purpose**: Checks how well the model learned.
    *   **How it works**: Calls `model.predict(X_test)` to get predictions on unseen data. It then compares these predictions against the actual answers (`y_test`) to calculate Accuracy, Precision, Recall, and F1-Score.
*   `predict_loan(model, customer_data)`:
    *   **Purpose**: Makes a prediction for a single new customer.
    *   **How it works**: Takes the customer's details, formats them into a 2D NumPy array, and uses `model.predict()` to output a 1 or 0, and `model.predict_proba()` to output the probability percentage.

### Inside `train_model.py` (The Execution Script)
*   `main()`:
    *   **Purpose**: The main script that runs the entire pipeline from top to bottom. It loads the data, splits it, calls the training function, evaluates it, prints the confusion matrix, and finally saves the model using `joblib.dump()`.

---

## 6. Features Used
- **Monthly Income** (Higher is better)
- **Credit Score** (Higher is better)
- **Employment Years** (Higher is better)
- **Monthly Expenses** (Lower is better)
- **Existing Loans**
- **Previous Loan Count**
- **Previous Loan Defaults** (Lower is better)
- **Loan Amount** (Lower risk if smaller compared to income)
- **Loan Term**

## 7. User Interface (GUI)
The user interface is built using **Streamlit** (`app.py`). Streamlit is an excellent Python library that turns data scripts into interactive web apps instantly. It allows the user to input customer details via text boxes and sliders, and instantly receive a prediction, probability scores, and view the Decision Tree visually.

## 8. How to Run the Project

1. **Install Requirements**: Ensure you have all the necessary Python libraries.
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**: Run the training script. This creates the dataset, trains the model, and saves `loan_model.pkl`.
   ```bash
   python train_model.py
   ```

3. **Run the Streamlit Web Application (GUI)**: Start the graphical user interface.
   ```bash
   streamlit run app.py
   ```
   *(Alternatively, if you prefer the terminal, you can run `python predict.py` for a text-based interface).*
