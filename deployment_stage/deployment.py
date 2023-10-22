import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# Load the pre-trained machine learning model
with open('best_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Create a Streamlit app
st.title("Welcome to FIFA Player Rating Prediction")
st.write("Enter player attributes to predict their overall rating")

# Define input fields for user input
potential = st.number_input("potential", min_value=0, max_value=100, value=50)
value_eur = st.number_input("value_eur", min_value=0, max_value=100, value=50)
wage_eur = st.number_input("wage_eur", min_value=0, max_value=100, value=50)
age = st.number_input("age", min_value=0, max_value=100, value=50)
weight_kg = st.number_input("weight_kg", min_value=0, value=0)
international_reputation = st.number_input("international_reputation", min_value=0, max_value=100, value=50)
shooting = st.number_input("shooting", min_value=0, value=0)
passing = st.number_input("passing", min_value=0, max_value=100, value=50)
dribbling = st.number_input("dribbling", min_value=0, value=0)
defending = st.number_input("defending", min_value=0, max_value=100, value=50)
physic = st.number_input("physic", min_value=0, max_value=100, value=50)
attacking_short_passing = st.number_input("Attacking Short Passing", min_value=0, max_value=100, value=50)
movement_reactions = st.number_input("movement_reactions", min_value=0, max_value=100, value=50)
power_shot_power = st.number_input("power_shot_power", min_value=0, max_value=100, value=50)
mentality_vision = st.number_input("mentality_vision", min_value=0, max_value=100, value=50)
mentality_composure = st.number_input("mentality_composure", min_value=0, max_value=100, value=50)

# Create a button to trigger the prediction
if st.button("Predict Overall Rating"):
    # Prepare the input data for prediction
    input_data = {
        "potential": potential,
        "value_eur": value_eur,
        "wage_eur": wage_eur,
        "age": age,
        "weight_kg": weight_kg,
        "international_reputation": international_reputation,
        "shooting": shooting,
        "passing": passing,
        "dribbling": dribbling,
        "defending": defending,
        "physic": physic,
        "attacking_short_passing": attacking_short_passing,
        "movement_reactions": movement_reactions,
        "power_shot_power":power_shot_power,
        "mentality_vision": mentality_vision,
        "mentality_composure": mentality_composure
    }
    input_df = pd.DataFrame([input_data])
    scaled_input_data = scaler.transform(input_df)

    # Make the prediction
    predicted_rating = loaded_model.predict(scaled_input_data)
    # Display the prediction
    st.write(f"Predicted Overall Rating: {predicted_rating[0]:.2f}")

# Create a reset button to clear the input fields
if st.button("Reset"):
    st.experimental_rerun()