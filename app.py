from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    rm = float(request.form['rm'])
    ptratio = float(request.form['ptratio'])
    lstat = float(request.form['lstat'])

    # Load the trained model
    with open('regression_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Create a DataFrame with the user input
    data = {'Unnamed: 0': [0], 'age': [0], 'black': [0], 'chas': [0], 'crim': [0], 'dis': [0],
            'indus': [0], 'lstat': [lstat], 'nox': [0], 'ptratio': [ptratio], 'rad': [0], 'rm': [rm],
            'tax': [0], 'zn': [0]}
    input_df = pd.DataFrame(data)

    # Make the prediction
    prediction = model.predict(input_df)

    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

