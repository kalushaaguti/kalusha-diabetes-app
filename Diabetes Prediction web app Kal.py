# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 19:58:10 2025

@author: kalus
"""

import numpy as np
import pickle
import streamlit as st

st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Prediction function
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]

# Main app
def main():
    # Sidebar
    st.sidebar.title("Diabetes Prediction Input")
    st.sidebar.markdown("Enter patient details below:")

    # User inputs
    Pregnancies = st.sidebar.text_input('Number of Pregnancies')
    Glucose = st.sidebar.text_input('Glucose Level')
    BloodPressure = st.sidebar.text_input('Blood Pressure')
    SkinThickness = st.sidebar.text_input('Skin Thickness')
    Insulin = st.sidebar.text_input('Insulin Level')
    BMI = st.sidebar.text_input('BMI')
    DiabetesPedigreeFunction = st.sidebar.text_input('Diabetes Pedigree Function')
    Age = st.sidebar.text_input('Age')

    # App title and description
    st.title("Diabetes Prediction Web App")
    st.write("This app uses a trained machine learning model to predict whether a person is **diabetic** or **non-diabetic** based on their health data.")

    diagnosis = ''
    
    # Prediction button
    if st.button('Predict Diabetes Status'):
        try:
            # Validate input (ensure all fields are filled)
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            if any(val.strip() == '' for val in input_data):
                st.warning("Please fill in all input fields before predicting.")
            else:
                result = diabetes_prediction(input_data)
                if result == 0:
                    st.success("The person is **Non-Diabetic**.")
                else:
                    st.error("The person is **Diabetic**.")
        except Exception as e:
            st.error(f"Error: {e}")

    # Footer
    st.markdown("---")
    st.caption("Developed by **Kalusha Aguti** | Machine Learning Engineer | SIUE")

if __name__ == '__main__':
    main()


