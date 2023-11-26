# app.py
from flask import Flask,render_template, request, jsonify
from models.randomforest_model import load_model
import numpy as np

app = Flask(__name__, template_folder='./templates')
randomforest_model = load_model()  # Load the model outside the route function


@app.route('/')
def home():
   return render_template('app.html')  # Adjust the path based on your folder structure

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()

        # Extract features from the input data
        features = [
            data['age'], 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex_0', 'sex_1',
            'cp_0', 'cp_1', ..., 'thal_0', 'thal_1', 'thal_2', 'thal_3'
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
