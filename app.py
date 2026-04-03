import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model and scaler
try:
    model = pickle.load(open('housepred.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError as e:
    raise RuntimeError(f"Model file not found: {e}")


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        if not data or 'data' not in data:
            return jsonify({'error': 'Invalid input. Expected JSON with "data" key.'}), 400
        values = np.array(list(data['data'].values())).reshape(1, -1)
        scaled_data = scaler.transform(values)
        output = model.predict(scaled_data)
        return jsonify({'prediction': float(output[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_values = request.form.values()
        data = [float(x) for x in form_values]
        if not data:
            return render_template('home.html', prediction_text="No input provided.")
        final_input = scaler.transform(np.array(data).reshape(1, -1))
        output = model.predict(final_input)[0]
        return render_template(
            'home.html',
            prediction_text="The House Price Prediction is: ${:,.2f}".format(float(output) * 1000)
        )
    except ValueError:
        return render_template('home.html', prediction_text="Invalid input. Please enter numeric values.")
    except Exception as e:
        return render_template('home.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True) 