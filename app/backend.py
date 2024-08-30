from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
with open('../model/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    features = np.array(data).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'class': 'drone' if prediction[0] == 1 else 'bird'})

if __name__ == '__main__':
    app.run(debug=True)
