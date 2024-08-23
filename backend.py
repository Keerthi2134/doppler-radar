from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the trained model
with open('model.py', 'rb') as f:
    model = pickle.load(f)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    # Assuming data is a list of features: [mean_frequency, energy]
    features = np.array(data).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'class': 'drone' if prediction[0] == 1 else 'bird'})

if __name__ == '__main__':
    app.run(debug=True)
