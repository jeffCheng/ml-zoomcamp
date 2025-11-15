import pickle

from flask import Flask
from flask import request
import pandas as pd
from sklearn.preprocessing import LabelEncoder

model_file = 'best_clf.bin'

with open(model_file, 'rb') as f_in:
    best_clf = pickle.load(f_in)

app = Flask('car_price')

# Encode categorical variables
categorical_columns = ['price_category','fuel_type', 'body_type', 'gearbox','brand','price_category']
label_encoders = {}

# Function to predict price category for a new car
def predict_car_price_category(car_features):
    # Convert the input features to match the training data format
    car_df = pd.DataFrame([car_features])

    # Encode categorical variables
    for column in categorical_columns:  # Exclude Price_Category
        if column in car_df.columns:
            label_encoders[column] = LabelEncoder()
            car_df[column] = label_encoders[column].fit_transform(car_df[column])
    
    
    # Make prediction
    prediction = best_clf.predict(car_df)[0]
    
    # Decode prediction
    return label_encoders['price_category'].inverse_transform([prediction])[0]

@app.route('/predict', methods=['POST'])
def predict():
    example_car = request.get_json()
    predicted_category = predict_car_price_category(example_car)
    repsonse = "Predicted price category for the example car: {predicted_category}"
    return repsonse

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)