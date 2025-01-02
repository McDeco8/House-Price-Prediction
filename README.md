"""
# Home Loan Default Prediction

## Project Overview
The Home Loan Default Prediction project aims to predict whether a customer will default on a home loan using historical data. This includes preparing a comprehensive data analysis report and building a predictive model to identify key factors or customer segments eligible for loans.

## Problem Statement
The main goals of this project are:
1. Prepare a complete data analysis report based on the given data.
2. Create a predictive model to identify factors or customer segments eligible for loans.

## Dataset Description
The dataset consists of multiple files containing information about applicants and their credit history:

- **`application_train.csv`**: Contains the target variable (1: Defaulter; 0: Not Defaulter) and static data for all applications. Each row represents one loan.
- **`bureau.csv`**: Data on all client's previous credits from other financial institutions, reported to the Credit Bureau.
- **`bureau_balance.csv`**: Monthly balances of previous credits in the Credit Bureau.
- **`POS_CASH_balance.csv`**: Monthly snapshots of previous POS and cash loans with Home Credit.
- **`credit_card_balance.csv`**: Monthly snapshots of previous credit cards with Home Credit.
- **`previous_application.csv`**: All previous applications for loans by clients in the sample.
- **`installments_payments.csv`**: Repayment history for previously disbursed credits.

## Files in the Repository

- **`Home_Loan_Default_final.ipynb`**: Jupyter notebook containing data analysis, model training, and evaluation.
- **`logistic_model.pkl`**: Serialized logistic regression model for deployment.
- **`metrics.json`**: JSON file containing performance metrics of the model.
- **`requirements.txt`**: Python dependencies required to run the project.
- **`results/`**: Directory for storing outputs such as predictions and reports.

## Installation
To set up the environment and run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
2. Install dependencies:
      ```bash
   pip install -r requirements.txt
4. Open the Jupyter notebook:
      ```bash
   jupyter notebook Home_Loan_Default_final.ipynb


## Usage

1. Run the notebook to preprocess the data, train the model, and evaluate its performance.
2. Use the **`logistic_model.pkl`** file to make predictions on new data:
     ```python 
      import pickle
      import numpy as np

      # Load the model
     with open('logistic_model.pkl', 'rb') as f:
          model = pickle.load(f)

     # Example prediction
      input_data = np.array([[...]])  # Replace with appropriate input features
      prediction = model.predict(input_data)
      print("Prediction:", prediction)

## Results
The logistic regression model achieved the following performance metrics:


The performance of the model on the test data is as follows:

| Metric                | Class 0 (Non-Defaulter)     |Class 1 (Defaulter)   |
|-----------------------|----------------------------|---------------------|
| Precision            | 0.92         |	0.18  | 
| Recall  | 0.99     |0.02     |
| F1-Score  |0.96   |0.03 | 



  Accuracy: 91%

  Macro Average: Precision = 0.55, Recall = 0.50, F1-Score = 0.49

  Weighted Average: Precision = 0.86, Recall = 0.91, F1-Score = 0.88



##  Challenges Faced
Data Imbalance: The dataset had significantly more non-defaulters than defaulters. Techniques such as oversampling or undersampling were considered to address this.

High Cardinality: Certain features had high cardinality, requiring dimensionality reduction techniques.

Correlated Features: Features with high correlation were identified and removed to avoid multicollinearity.
