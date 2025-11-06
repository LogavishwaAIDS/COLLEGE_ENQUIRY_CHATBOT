from flask import Flask, render_template, jsonify, request
from chat import get_response
import os
import nltk

nltk.download('punkt')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('base.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json().get("message")
    response = get_response(data)
    return jsonify({"answer": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
