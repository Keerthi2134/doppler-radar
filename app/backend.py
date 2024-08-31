from flask import Flask, request, jsonify
import numpy as np
import pickle
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

# Load the trained model
model_path = '../model/model.pkl'
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    logging.error(f"Model file {model_path} not found.")
    model = None
except pickle.PickleError:
    logging.error("Error loading the model.")
    model = None

@app.route('/classify', methods=['POST'])
def classify():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    if not isinstance(data, list) or len(data) != 5000:
        return jsonify({'error': 'Invalid input data format'}), 400
    
    features = np.array(data).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'class': 'drone' if prediction[0] == 1 else 'bird'})

if __name__ == '__main__':
    app.run(debug=True)
