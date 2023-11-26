# app.py
from flask import Flask, request, jsonify
from models.randomforest_model import load_model

app = Flask(__name__)
models = joblib.load('randomforest_model.joblib')

# Print the keys and the type of each item in the loaded models dictionary
for key, value in models.items():
    print(f"Key: {key}, Type: {type(value)}")

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

        # Use the loaded KNN model for predictions
        knn_prediction = knn_model.predict(features_array)

        # Use the loaded Random Forest model for predictions
        randomforest_prediction = randomforest_model.predict(features_array)

        # Return the predictions
        return jsonify({'knn_prediction': int(knn_prediction[0]), 'randomforest_prediction': int(randomforest_prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)