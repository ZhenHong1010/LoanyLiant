from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('best_rf_model.pkl', 'rb') as pkl_file:
    best_rf_model = pickle.load(pkl_file)

# Define the prediction function
def predict(input_features):
    input_array = np.array([input_features])
    prediction = best_rf_model.predict(input_array)
    return 'YES' if prediction == 1 else 'NO'

# Function to label encode the loan purpose
def encode_loan_purpose(loan_purpose):
    loan_purpose_map = {
        'business': 0,
        'emergency_funds': 1,
        'home': 2,
        'investment': 3,
        'other': 4
    }
    return loan_purpose_map[loan_purpose]

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the loan prediction page
@app.route('/loan_predictor', methods=['GET', 'POST'])
def loan_predictor():
    if request.method == 'POST':
        # Get form data
        saving_amount = float(request.form['saving_amount'])
        checking_amount = float(request.form['checking_amount'])
        yearly_salary = float(request.form['yearly_salary'])
        total_credit_card_limit = float(request.form['total_credit_card_limit'])
        currently_repaying_other_loans = request.form['currently_repaying_other_loans']
        is_employed = request.form['is_employed']
        avg_percentage_credit_card_limit_used_last_year = float(request.form['avg_percentage_credit_card_limit_used_last_year'])
        dependent_number = int(request.form['dependent_number'])
        age = int(request.form['age'])
        loan_purpose_cat = request.form['loan_purpose_cat']
        fully_repaid_previous_loans = request.form['fully_repaid_previous_loans']
        
        # Label encode the loan purpose
        loan_purpose_encoded = encode_loan_purpose(loan_purpose_cat)
        
        # Process inputs into the format expected by the model
        input_features = [
            saving_amount,
            checking_amount,
            yearly_salary,
            total_credit_card_limit,
            1 if currently_repaying_other_loans == 'yes' else 0,
            1 if is_employed == 'yes' else 0,
            avg_percentage_credit_card_limit_used_last_year,
            dependent_number,
            age,
            loan_purpose_encoded,
            1 if fully_repaid_previous_loans == 'yes' else 0
        ]
        
        # Make prediction
        result = predict(input_features)
        
        # Flip the result for display
        display_result = 'NO' if result == 'YES' else 'YES'
        
        # Render predict.html with the flipped result
        return render_template('predict.html', result=display_result)
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
