# Step 1: Install required libraries (if not already installed)
# pip install pandas scikit-learn streamlit

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import streamlit as st

# Step 2: Load the dataset and preprocess it
# Replace 'your_dataset.csv' with the actual file path of your dataset
dataset_path = 'Heart_Disease.csv'
df = pd.read_csv(dataset_path)

# Drop rows with missing values (optional)
df = df.dropna()

# Separate features (X) and target (y)
X = df.drop('TenYearCHD', axis=1)
y = df['TenYearCHD']

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train the heart disease prediction model (Logistic Regression)
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_scaled, y)

# Save the trained model to a file (pickle format)
import pickle
model_file = 'Umodel.pkl'
with open(model_file, 'wb') as file:
    pickle.dump(model, file)

# Step 4: Create a Streamlit web app
def predict_heart_disease(input_data):
    # Convert input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Scale the input features using the same scaler as used during training
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    return prediction[0]

def main():
    st.title('Heart Disease Prediction')
    st.write('Enter the following details to predict the risk of heart disease within the next ten years.')

    # Collect user input
    # Replace 'Attribute Name' with the actual attribute names and set appropriate min_value and max_value
    male = st.selectbox('Gender:', df['male'].unique())
    age = st.number_input('Age:', min_value=1, max_value=100)
    currentSmoker = st.selectbox('Current Smoker:', df['currentSmoker'].unique())
    cigsPerDay = st.number_input('Cigarettes per day:', min_value=0, max_value=50)
    BPMeds = st.selectbox('BP Medication:', df['BPMeds'].unique())
    prevalentStroke = st.selectbox('Prevalent Stroke:', df['prevalentStroke'].unique())
    prevalentHyp = st.selectbox('Prevalent Hypertension:', df['prevalentHyp'].unique())
    diabetes = st.selectbox('Diabetes:', df['diabetes'].unique())
    totChol = st.number_input('Total Cholesterol (mg/dL):', min_value=100, max_value=600)
    sysBP = st.number_input('Systolic Blood Pressure (mmHg):', min_value=70, max_value=250)
    diaBP = st.number_input('Diastolic Blood Pressure (mmHg):', min_value=40, max_value=150)
    BMI = st.number_input('BMI:', min_value=10, max_value=50)
    heartRate = st.number_input('Resting Heart Rate (bpm):', min_value=40, max_value=150)
    glucose = st.number_input('Glucose (mg/dL):', min_value=50, max_value=400)

    # Make prediction
    if st.button('Predict'):
        input_data = {
            'male': male,
            'age': age,
            'currentSmoker': currentSmoker,
            'cigsPerDay': cigsPerDay,
            'BPMeds': BPMeds,
            'prevalentStroke': prevalentStroke,
            'prevalentHyp': prevalentHyp,
            'diabetes': diabetes,
            'totChol': totChol,
            'sysBP': sysBP,
            'diaBP': diaBP,
            'BMI': BMI,
            'heartRate': heartRate,
            'glucose': glucose
        }
        prediction = predict_heart_disease(input_data)

        if prediction == 0:
            st.write("Congratulations! You are at a low risk of heart disease.")
        else:
            st.write("You are at a higher risk of heart disease. Please consult a doctor for further evaluation.")

if __name__ == '__main__':
    main()
