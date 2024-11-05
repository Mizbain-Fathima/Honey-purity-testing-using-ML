import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Path to honey purity model file
model_path = r'C:\Users\Lenovo\Desktop\ML-honey-mam\honey_purity_rf_model.pkl'

# Input features 
input_features = {
    'CS': 1.23, 
    'Density': 0.95,
    'WC': 18.5,
    'pH': 4.5,
    'EC': 1.02,
    'F': 34.1,
    'G': 45.0,
    'Pollen_analysis': 'Alfalfa',
    'Viscosity': 1200.0,
    'Price': 5.6
}

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_input(input_data, data):
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode categorical features
    input_df = pd.get_dummies(input_df, columns=['Pollen_analysis'], prefix='', prefix_sep='')
    
    # Ensure all expected columns are present
    unique_pollen_categories = data['Pollen_analysis'].unique()
    expected_columns = ['CS', 'Density', 'WC', 'pH', 'EC', 'F', 'G', 'Viscosity', 'Price'] + list(unique_pollen_categories)
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns to match model's expected input
    input_df = input_df[expected_columns]
    
    # Scale the input features
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(input_df)
    
    return scaled_input

def make_prediction(model, scaled_input):
    prediction = model.predict(scaled_input)
    prediction_prob = model.predict_proba(scaled_input)
    return prediction[0], prediction_prob[0][1]
    
if __name__ == "__main__":
    try:
        data = pd.read_csv(r'C:\Users\Lenovo\Desktop\ML-honey-mam\dataset\honey_purity_dataset.csv')
        model = load_model(model_path)
        scaled_input = preprocess_input(input_features, data)
        prediction, probability = make_prediction(model, scaled_input)
        if prediction == 1:
            print("The honey is predicted to be pure.")
        else:
            print("The honey is predicted to be impure.")
        
        print(f"Probability of purity: {probability:.2f}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
