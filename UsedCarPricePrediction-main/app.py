import numpy as np
import pickle
import pandas as pd
import streamlit as st

# Load the model and scaler
try:
    with open('random_forest_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except FileNotFoundError as e:
    st.error("Model file not found. Please ensure it's present in the directory.")


# Define the prediction function
def predict_price(year, showroom_price, kilometer_driven, owner_count, fuel_type, transmission):
    # Create a DataFrame for the input features
    features = pd.DataFrame([[year, showroom_price, kilometer_driven, owner_count, fuel_type, transmission]],
                            columns=['year', 'showroom_price', 'kilometer_driven', 'owner_count', 'fuel_type', 'transmission'])
    
    # One-hot encode the categorical variables
    features = pd.get_dummies(features, columns=['fuel_type', 'transmission'])

    # Ensure all required columns are present
    required_columns = ['year', 'showroom_price', 'kilometer_driven', 'owner_count', 
                        'fuel_type_Diesel', 'fuel_type_Petrol',  
                        'transmission_Manual', 'transmission_Automatic']
    for col in required_columns:
        if col not in features.columns:
            features[col] = 0
    
    # Align the columns
    features = features[required_columns]

    # Predict the price
    prediction = model.predict(features)

    return prediction[0]

# Streamlit app
st.title("Used Car Price Prediction")

# Input fields
year = st.number_input("Year", min_value=1980, max_value=2024, value=2015)
showroom_price = st.number_input("Showroom Price (in lakhs)", min_value=0, value=500000)
kilometer_driven = st.number_input("Kilometer Driven", min_value=0, value=40000)
owner_count = st.number_input("Number of Previous Owners", min_value=0, value=1)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# Predict button
if st.button("Predict Price"):
    predicted_price = predict_price(year, showroom_price, kilometer_driven, owner_count, fuel_type, transmission)
    st.success(f"The predicted price of the car is ₹{predicted_price:,.2f}")

if __name__ == '__main__':
    st.write("Run the script using: `streamlit run app.py`")
