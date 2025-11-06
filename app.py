from flask import Flask, render_template, jsonify, request
from chat import get_response
import os
import nltk

# Download nltk data
nltk.download('punkt')

app = Flask(__name__)

# ---- Routes ----
@app.route('/')
def home():
    return render_template('base.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json().get("message")
    response = get_response(data)
    message = {"answer": response}
    return jsonify(message)

# ---- Main Entry ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
