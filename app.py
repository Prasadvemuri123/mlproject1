import streamlit as st
import pickle
import numpy as np
import bz2

# Load the trained model from the compressed .bz2 file
def load_model(file_path):
    with bz2.open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Path to the compressed model file
model = load_model('C:\\Users\\prasa\\rainfall_prediction_model_compressed (1).pkl.bz2')

# Define a function for prediction
def predict_rainfall(input_data):
    prediction = model.predict(input_data)
    return "It will rain tomorrow." if prediction[0] == 1 else "It will not rain tomorrow."

# Title and description
st.title("Rainfall Prediction App")
st.write("Enter weather parameters below to predict if it will rain tomorrow.")

# Collect user input for each feature in the model
def get_user_input():
    min_temp = st.number_input("Min Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0)
    max_temp = st.number_input("Max Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=5.0)
    evaporation = st.number_input("Evaporation (mm)", min_value=0.0, max_value=500.0, value=2.0)
    sunshine = st.number_input("Sunshine (hours)", min_value=0.0, max_value=24.0, value=8.0)
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=200.0, value=15.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
    pressure = st.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0)
    cloud = st.number_input("Cloud Cover (1-8 scale)", min_value=0.0, max_value=8.0, value=4.0)
    temp_9am = st.number_input("Temperature at 9am (°C)", min_value=-10.0, max_value=50.0, value=18.0)
    humidity_3pm = st.number_input("Humidity at 3pm (%)", min_value=0.0, max_value=100.0, value=50.0)

    # Gather inputs in the correct order as a numpy array
    user_input = np.array([[min_temp, max_temp, rainfall, evaporation, sunshine, wind_speed,
                            humidity, pressure, cloud, temp_9am, humidity_3pm]])
    return user_input

# User input section
user_input = get_user_input()

# Predict button
if st.button("Predict"):
    result = predict_rainfall(user_input)
    st.write(result)
