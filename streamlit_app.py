import streamlit as st 
import pandas as pd 
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler, LabelEncoder




# Load model, scaler, and encoder
def load_model():
    with open("student_lr_final_model.pkl", "rb") as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le
# Prepare user input
def preprocessing_input_data(data, scaler, le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed
# Make prediction
def predict_data(data):
    model, scaler, le = load_model()
    processed_data = preprocessing_input_data(data, scaler, le)
    prediction = model.predict(processed_data)
    return prediction
# Streamlit interface
def main():
    st.title("🎓 Student Performance Prediction App")
    st.write("Fill in the details to predict your academic performance score")
    hour_studied = st.number_input("Hours Studied", min_value=1, max_value=10, value=5)
    previous_score = st.number_input("Previous Score", min_value=40, max_value=100, value=70)
    sleeping_hour = st.number_input("Sleeping Hours", min_value=4, max_value=10, value=7)
    number_of_paper_solved = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=10, value=5)
    extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])
   
   
    if st.button("Predict Your Score"):
        user_data = {
            "Hours Studied": hour_studied,
            "Previous Scores": previous_score,
            "Sleep Hours": sleeping_hour,
            "Sample Question Papers Practiced": number_of_paper_solved,
            "Extracurricular Activities": extra
        }
        prediction = predict_data(user_data)
        st.success(f"📊 Your Predicted Performance Score is: {prediction[0][0]:.2f}")
if __name__ == "__main__":
    main()
