import pickle

from flask import Flask
from flask import request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from flask import jsonify


model_file = 'best_regressor.bin'

with open(model_file, 'rb') as f_in:
    best_regressor = pickle.load(f_in)

app = Flask('car_price')
scaler = StandardScaler()

def predict_car_price(car_features):
    # Convert the input features to match the training data format
    car_df = pd.DataFrame([car_features])
    # Scale features
    car_scaled = scaler.fit_transform(car_df)
    
    # Make prediction
    return best_regressor.predict(car_scaled)[0]

@app.route('/predict', methods=['POST'])
def predict():
    example_car = request.get_json()
    predicted_price = predict_car_price(example_car)
    result = {
        'price': predicted_price
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)