# app.py
from flask import Flask,render_template, request, jsonify
from models.randomforest_model import load_model
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='./templates')
randomforest_model = load_model()  # Load the model outside the route function


dataset_path = './models/dataset.csv'
data = pd.read_csv(dataset_path)
X_train = data.drop('target', axis=1)  # Adjust the column name according to your dataset
y_train = data['target']  # Adjust the column name according to your dataset
randomforest_model.fit(X_train, y_train)


@app.route('/')
def home():
   return render_template('app.html')  # Adjust the path based on your folder structure

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()

        # Convert relevant features to float
        features = [
            float(data['age']), float(data['trestbps']), float(data['chol']),
            float(data['thalach']), float(data['oldpeak']), float(data['sex_0']),
            float(data['sex_1']), float(data['cp_0']), float(data['cp_1']),
            float(data['thal_0']), float(data['thal_1']), float(data['thal_2']), float(data['thal_3'])
        ]

        # Convert features to a NumPy array for prediction
        features_array = np.array([features], dtype=np.float64)

        # Use the loaded Random Forest model for predictions
        prediction = randomforest_model.predict(features_array)

        # Return the prediction
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)