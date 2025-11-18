#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle


df = pd.read_csv('used_cars_UK.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)
df.columns = map(str.lower, df.columns)
df.rename(columns={'mileage(miles)': 'mileage'}, inplace=True)
df.rename(columns={'previous owners': 'previous_owners'}, inplace=True)
df.rename(columns={'fuel type': 'fuel_type'}, inplace=True)
df.rename(columns={'body type': 'body_type'}, inplace=True)
df.rename(columns={'emission class': 'emission_class'}, inplace=True)
df.rename(columns={'service history': 'service_history'}, inplace=True)

df[['brand', 'car_name']] = df['title'].str.split(' ', n=1, expand=True)

df['service_history'] = df['service_history'].fillna(0)
df['service_history'] = df['service_history'].replace('Full', 1)
df['service_history']= df[['service_history']].astype('bool')
df['engine'] = df['engine'].str.replace('L', '', regex=False)
df['engine'] = pd.to_numeric(df['engine'], errors='coerce')

df['previous_owners'] = df['previous_owners'].fillna(0)

df['doors'] = df['doors'].fillna(0)
df['seats'] = df['seats'].fillna(0)

categorical_columns = ['fuel_type', 'body_type', 'gearbox','brand','price_category']
label_encoders = {}

for column in categorical_columns:
    if column in df.columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

# Prepare features and target
X = df.drop(['price', 'title','emission_class','car_name','fuel_type', 'body_type', 'gearbox','brand'], axis=1)
y = df['price']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Split the data into train, validation, and test sets (60%, 20%, 20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Create and train decision trees with different parameters
max_depths = [3, 5, 7, 10, 15, None]
min_samples_splits = [2, 5, 10]
results = []

for depth in max_depths:
    for min_samples_split in min_samples_splits:
        # Create and train the model
        regressor = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        regressor.fit(X_train_scaled, y_train)
        
        # Make predictions on validation set
        y_val_pred = regressor.predict(X_val_scaled)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        results.append({
            'max_depth': depth,
            'min_samples_split': min_samples_split,
            'val_mse': val_mse,
            'val_r2': val_r2,
            'model': regressor
        })
        
        print(f"\nDecision Tree (max_depth={depth}, min_samples_split={min_samples_split}):")
        print(f"Validation MSE: {val_mse:.2f}")
        print(f"Validation R2: {val_r2:.4f}")

# Find best model based on validation MSE
best_result = min(results, key=lambda x: x['val_mse'])
best_regressor = best_result['model']

print(f"\nBest model parameters:")
print(f"Max depth: {best_result['max_depth']}")
print(f"Min samples split: {best_result['min_samples_split']}")
print(f"Validation MSE: {best_result['val_mse']:.2f}")
print(f"Validation R2: {best_result['val_r2']:.4f}")

# Evaluate best model on test set
y_test_pred = best_regressor.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"\nTest Set Performance:")
print(f"MSE: {test_mse:.2f}")
print(f"MAE: {test_mae:.2f}")
print(f"R2: {test_r2:.4f}")

# Function to predict price for a new car
def predict_car_price(car_features):
    # Convert the input features to match the training data format
    car_df = pd.DataFrame([car_features])
    # Encode categorical variables
    for column in categorical_columns:
        if column in car_df.columns:
            car_df[column] = label_encoders[column].transform(car_df[column])
    
    # Scale features
    car_scaled = scaler.transform(car_df)
    
    # Make prediction
    return best_regressor.predict(car_scaled)[0]

# Example usage
example_car_1 = {
    'mileage': 12203,
    'registration_year': 2009,
    'previous_owners': 1,
    'engine': 2.0,
    'doors': 5,
    'seats': 5,
    'service_history': True
}
predicted_price = predict_car_price(example_car_1)
print(f"\nPredicted price for the example car: £{predicted_price:.2f}")


output_file = f'best_regressor.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump(best_regressor, f_out)

print(f'the model is saved to {output_file}')