from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import seaborn as sns
import streamlit as st
# Your Flask app code here


app = Flask(__name__)

# Load the trained model
with open('logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the scaler for the variables
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the label encoders for categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Occupation', 'BMI Category']
for col in categorical_columns:
    with open(f'label_encoder_{col}.pkl', 'rb') as f:
        label_encoders[col] = pickle.load(f)

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    elif request.method == 'POST':
        features = {
        'Gender': request.form['Gender'],
        'Age': int(request.form['Age']),
        'Occupation': request.form['Occupation'],
        'Sleep Duration': float(request.form['Sleep Duration']),
        'Quality of Sleep': int(request.form['Quality of Sleep']),
        'Physical Activity Level': int(request.form['Physical Activity Level']),
        'Stress Level': int(request.form['Stress Level']),
        'BMI Category': request.form['BMI Category'],
        'Heart Rate': int(request.form['Heart Rate']),
        'Daily Steps': int(request.form['Daily Steps']),
        'Systolic': int(request.form['Systolic']),
        'Diastolic': int(request.form['Diastolic']),
        }
    

        # Encode categorical features
        for col in categorical_columns:
            features[col] = label_encoders[col].transform([features[col]])[0]
    
        # Scale numerical features
        numerical_features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                          'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic']
    
        numerical_values = np.array([features[col] for col in numerical_features]).reshape(1, -1)
        scaled_values = scaler.transform(numerical_values)
    
        features['Age']=scaled_values[0][0]
        features['Sleep Duration']=scaled_values[0][1]
        features['Quality of Sleep']=scaled_values[0][2]
        features['Physical Activity Level']=scaled_values[0][3]
        features['Stress Level']=scaled_values[0][4]
        features['Heart Rate']=scaled_values[0][5]
        features['Daily Steps']=scaled_values[0][6]
        features['Systolic']=scaled_values[0][7]
        features['Diastolic']=scaled_values[0][8]
    
    
        input_values = list(features.values())

        # Convert the list to a 2D array with shape (1, num_features)
        input_array = np.array(input_values).reshape(1, -1)
        # input_values = list(features.values())
        # Make prediction
        prediction = model.predict(input_array)
        predicted_sleep_disorder = prediction[0]
        if(predicted_sleep_disorder==0):
            prediction_result='Insomnia'
        elif(predicted_sleep_disorder==2):
            prediction_result='Sleep Apnea'
        else:
            prediction_result='None'

        return render_template('result.html', predicted_sleep_disorder=prediction_result)


# Route for rendering the HTML template
@app.route('/about')
def index():
    return render_template('About.html')



# Define your Streamlit app code here

if __name__ == "__main__":
    try:
        # Run your Streamlit app
        app.run(debug=True)
    except Exception as e:
        # Handle any exceptions that occur during execution
        st.error("An error occurred: {}".format(str(e)))
