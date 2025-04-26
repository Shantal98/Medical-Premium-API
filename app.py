pip install flask
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Initialize the Flask app
app = Flask(__name__)

# --- Load Dataset ---
@st.cache_data
def load_data():
    url = 'https://github.com/Shantal98/Medical-Premium-API/blob/main/Medicalpremium.csv'
    df = pd.read_csv(url)
    
    # Calculate BMI
    df['BMI'] = df['Weight'] / ((df['Height'] / 100) ** 2)
    
    return df

df = load_data()

# --- Preprocessing ---
X = df.drop(columns=['PremiumPrice'])
y = df['PremiumPrice']

# Use BMI instead of Height and Weight
X['BMI'] = df['BMI']
X = X.drop(columns=['Height', 'Weight'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training (only first time or on update) ---
def train_model():
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train.columns.tolist()

model, model_features = train_model()

# --- Helper Functions ---
def calculate_bmi(weight, height):
    return weight / ((height / 100) ** 2)

def predict_premium(age, weight, height, diabetes, blood_pressure, transplant, chronic, allergies, cancer_history, surgeries):
    bmi = calculate_bmi(weight, height)
    
    # Simulated prediction for demo purposes
    input_data = pd.DataFrame({
        'Age': [age],
        'Weight': [weight],
        'Height': [height],
        'BMI': [bmi],
        'Diabetes': [1 if diabetes == "Yes" else 0],
        'BloodPressureProblems': [1 if blood_pressure == "Yes" else 0],
        'AnyTransplants': [1 if transplant == "Yes" else 0],
        'AnyChronicDiseases': [1 if chronic == "Yes" else 0],
        'KnownAllergies': [1 if allergies == "Yes" else 0],
        'HistoryOfCancerInFamily': [1 if cancer_history == "Yes" else 0],
        'NumberOfMajorSurgeries': [surgeries]
    })
    
    input_data = input_data.reindex(columns=model_features, fill_value=0)
    premium = model.predict(input_data)
    return premium[0]

# --- Define the API endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    age = data['age']
    weight = data['weight']
    height = data['height']
    diabetes = data['diabetes']
    blood_pressure = data['blood_pressure']
    transplant = data['transplant']
    chronic = data['chronic']
    allergies = data['allergies']
    cancer_history = data['cancer_history']
    surgeries = data['surgeries']

    premium = predict_premium(age, weight, height, diabetes, blood_pressure, transplant, chronic, allergies, cancer_history, surgeries)

    return jsonify({"predicted_premium": premium})

if __name__ == '__main__':
    app.run(debug=True)
