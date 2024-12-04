import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import joblib  # For loading the saved model

# Load pre-trained model and scalers
@st.cache
def load_model():
    model = joblib.load('pokemon_attack_predictor_model.pkl')  # Replace with your model path
    scaler = joblib.load('scaler.pkl')  # Replace with your scaler path
    encoder = joblib.load('encoder.pkl')  # Replace with your encoder path
    return model, scaler, encoder

# Prediction function
def predict_attack(model, scaler, encoder, input_data):
    # Scale numerical inputs
    numerical_features = ['HP', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    print(input_data)
    input_df = pd.DataFrame([input_data])  # Wrap the dictionary in a list to create a DataFrame
    scaled_data = scaler.transform(input_df[numerical_features])
    # scaled_data = scaler.transform(pd.DataFrame(input_data[numerical_features], columns=numerical_features))
    
    # Encode type inputs
    type_1 = input_data['Type 1']
    type_2 = input_data['Type 2']
    encoded_data = encoder.transform([[type_1, type_2]]).toarray()
    
    # Combine scaled numerical and encoded type data
    features = np.hstack((scaled_data, encoded_data))
    
    # Predict attack
    prediction = model.predict(features)
    return prediction[0]

# Load model and scalers
model, scaler, encoder = load_model()

# Streamlit app
st.title("Pokémon Attack Predictor")
st.write("This app predicts a Pokémon's attack based on its stats and type.")

# User inputs
st.header("Input Pokémon Stats")
hp = st.number_input("HP", min_value=1, max_value=255, value=60)
defense = st.number_input("Defense", min_value=1, max_value=255, value=50)
special_attack = st.number_input("Special Attack", min_value=1, max_value=255, value=50)
special_defense = st.number_input("Special Defense", min_value=1, max_value=255, value=50)
speed = st.number_input("Speed", min_value=1, max_value=255, value=60)

st.header("Select Pokémon Type")
type_1 = st.selectbox("Primary Type", ["Normal", "Fire", "Water", "Grass", "Electric", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"])
type_2 = st.selectbox("Secondary Type", ["None", "Normal", "Fire", "Water", "Grass", "Electric", "Ice", "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug", "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy"])

# Prediction button
if st.button("Predict Attack"):
    # Prepare input data
    input_data = {
        'HP': hp,
        'Defense': defense,
        'Sp. Atk': special_attack,
        'Sp. Def': special_defense,
        'Speed': speed,
        'Type 1': type_1,
        'Type 2': type_2,
    }
    
    # Get prediction
    predicted_attack = predict_attack(model, scaler, encoder, input_data)
    
    # Display result
    st.success(f"The predicted Attack value is: {predicted_attack:.2f}")

