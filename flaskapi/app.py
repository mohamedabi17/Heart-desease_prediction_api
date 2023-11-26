# app.py
from flask import Flask
from sklearn.ensemble import RandomForestClassifier
from joblib import load

app = Flask(__name__)
# models/your_model_module.py

def load_model():
    # Load and return the trained RandomForestClassifier model
    model = load('randomforest_model.joblib')
    return model

@app.route('/')
def hello():
    return 'Hello, Flask!'

if __name__ == '__main__':
    app.run(debug=True)
