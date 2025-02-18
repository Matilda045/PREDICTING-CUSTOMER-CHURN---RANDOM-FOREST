
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from re import M

# load the file that contains the model (model.pkl)
with open("model.pkl", "rb") as f:
  model = pickle.load(f)

# give the Streamlit app page a title
st.title("Customer Churn Predictor")

# input widget for getting user values for X (feature matrix value)
Last_Login_Days = st.slider("DaysSinceLastLogin", min_value=0, max_value=1000, value=20)
Login_Freq = st.slider("LoginFrequency", min_value=0, max_value=1000, value=20)
Age = st.slider("Age", min_value=0, max_value=1000, value=20)
Last_Interaction_Days = st.slider("DaysSinceInteraction", min_value=0, max_value=1000, value=20)
Amount_Spent = st.slider("AmountSpent", min_value=0, max_value=1000, value=20)
Last_Transaction_Days = st.slider("DaysSinceTransaction", min_value=0, max_value=1000, value=20)
Gender = st.slider("Gender", min_value=0, max_value=1, value=2)
Income_Medium = st.slider("Income_Medium", min_value=0, max_value=1000, value=20)
Marital_Status_Widowed = st.slider("Marital_Widowed", min_value=0, max_value=1, value=2)
Marital_Status_Divorced = st.slider("Marital_Divorced", min_value=0, max_value=1, value=2)

input_data = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
# After selecting features
if st.button("Predict"):
  
  # Make prediction
  churn_probability = model.predict_proba(Last_Login_Days, Login_Freq, Age, Last_Interaction_Days, Amount_Spent, Last_Transaction_Days, Gender, Income_Medium, Marital_Status_Widowed, Marital_Status_Divorced)[0][1] # Probability of churn (class 1)
  churn_prediction = model.predict(Last_Login_Days, Login_Freq, Age, Last_Interaction_Days, Amount_Spent, Last_Transaction_Days, Gender, Income_Medium, Marital_Status_Widowed, Marital_Status_Divorced)[0] # 0 or 1 for no churn or churn
  st.write(f"The probability of this customer churning is {churn_probability:.2%}")

