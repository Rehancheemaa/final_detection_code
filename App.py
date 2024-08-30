# from flask import Flask, request, jsonify
# import joblib
# import numpy as np
#
# # Load the trained Naive Bayes model and CountVectorizer
# count_vect = joblib.load('count_vect.pkl')
# Naive_loaded = joblib.load('naive_bayes_model.pkl')
#
# # Initialize Flask app
# app = Flask(__name__)
#
# # Define a route for prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the text data from the POST request
#     data = request.get_json()
#     if 'text' not in data:
#         return jsonify({'error': 'No text provided'}), 400
#
#     text = data['text']
#
#     # Convert the text to the model's expected format
#     text_vec = count_vect.transform([text])
#
#     # Make a prediction
#     prediction = Naive_loaded.predict(text_vec)
#
#     # Map the prediction to a human-readable label
#     labels_map = {
#         0: 'non-bully',
#         1: 'bully',
#         3: 'suicide',
#         4: 'non-suicide',
#     }
#     prediction_label = labels_map.get(prediction[0], 'Unknown')
#
#     # Return the prediction as a JSON response
#     return jsonify({'prediction': prediction_label})
#
# if __name__ == '__main__':
#     app.run(host='localhost', port=8000)


from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained Naive Bayes model and CountVectorizer
count_vect = joblib.load('count_vect.pkl')
Naive_loaded = joblib.load('naive_bayes_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the text data from the POST request
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']

    # Convert the text to the model's expected format
    text_vec = count_vect.transform([text])

    # Make a prediction
    prediction = Naive_loaded.predict(text_vec)

    # Map the prediction to a human-readable label
    labels_map = {
        0: 'non-bully',
        1: 'bully',
        3: 'suicide',
        4: 'non-suicide',
    }
    prediction_label = labels_map.get(prediction[0], 'Unknown')

    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(host='localhost', port=8000)
