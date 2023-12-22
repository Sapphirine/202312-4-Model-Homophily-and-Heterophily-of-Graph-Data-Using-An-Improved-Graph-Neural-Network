from flask import Flask, request, jsonify
from inference import perform_inference
from flask_cors import CORS
import random


app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    dataset = request.json
    prediction = perform_inference(dataset)
    print(prediction)
    return prediction

if __name__ == '__main__':
    app.run()
