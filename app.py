from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import seaborn as sns

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


@app.route('/datasetDesc')
def descdataset():
    # Load your data and perform necessary preprocessing
    df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
    
    # Apply table styles to the first 10 rows
    styled_df = df.style.set_table_styles([
        {'selector': 'th', 'props': [('border', '1px solid #ddd'), ('font-size', '10pt')]},
        {'selector': 'td', 'props': [('border', '1px solid #ddd'), ('font-size', '10pt')]},
        {'selector': 'tr:nth-of-type(odd)', 'props': [('background-color', 'lightsteelblue')]}
    ])
    
    # Render the HTML representation of the styled DataFrame
    styled_html = styled_df.render()
    
    return render_template('dataset.html', styled_html=styled_html)


# Load your data
data = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')  

# Define your plotting functions
def plot_categorical_distribution():
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    for i, var in enumerate(['Gender', 'BMI Category', 'Sleep Disorder']):
        col = i
        order = data[var].value_counts().index
        sns.countplot(x=var, data=data, ax=axes[col], order=order)    
        axes[col].set_title(f'Distribution of {var}', fontsize=14)
        axes[col].set_xlabel(var, fontsize=12)
        axes[col].set_ylabel('Count', fontsize=12)
        axes[col].set_xticklabels(axes[col].get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig('static/categorical_distribution.png')

def plot_occupational_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    order = data['Occupation'].value_counts().index
    sns.countplot(x='Occupation', data=data, ax=ax1, order=order)
    ax1.set_title('Distribution of Occupations', fontsize=14)
    ax1.set_xlabel('Occupation', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    
    occup_dis = data.groupby('Occupation')['Sleep Disorder'].value_counts(normalize=True).unstack().sort_values(by='None', ascending=False)
    order_sleep_disorder = ['None', 'Sleep Apnea', 'Insomnia']
    occup_dis[order_sleep_disorder].plot(kind='barh', stacked=True, ax=ax2)
    ax2.set_title('Sleep Disorder per Occupation', fontsize=14)
    ax2.set_xlabel('Proportions', fontsize=12)
    ax2.set_ylabel('Occupation', fontsize=12)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.savefig('static/occupational_distribution.png')

def plot_sleep_duration_boxplot():
    plt.figure(figsize=(10, 5)) 
    sns.boxplot(data=data,y='Sleep Duration', x='Occupation')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('static/sleep_duration_boxplot.png')

# Route for rendering the HTML template
@app.route('/plots')
def index():
    plot_categorical_distribution()
    plot_occupational_distribution()
    plot_sleep_duration_boxplot()
    return render_template('plottings.html')

import streamlit as st

# Define your Streamlit app code here

if __name__ == "__main__":
    try:
        # Run your Streamlit app
        app.run(debug=True)
    except Exception as e:
        # Handle any exceptions that occur during execution
        st.error("An error occurred: {}".format(str(e)))
