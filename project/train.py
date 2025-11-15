#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
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

# Create price categories
def categorize_price(price):
    if price <= 5000:
        return 'Low'
    elif price <= 10000:
        return 'Medium'
    elif price <= 15000:
        return 'High'
    else:
        return 'Very High'

df['price_category'] = df['price'].apply(categorize_price)

# Encode categorical variables
categorical_columns = ['fuel_type', 'body_type', 'gearbox','brand','price_category']
label_encoders = {}

for column in categorical_columns:
    if column in df.columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

# Prepare features and target
X = df.drop(['price', 'price_category', 'title','emission_class','car_name'], axis=1)
y = df['price_category']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create and train decision trees with different parameters
max_depths = [3, 5, 7, 10]
criteria = ['gini', 'entropy']
results = []

for depth in max_depths:
    for criterion in criteria:
        # Create and train the model
        clf = DecisionTreeClassifier(max_depth=depth, criterion=criterion, random_state=42)
        clf.fit(X_train, y_train)
        
        # Make predictions on validation set
        y_val_pred = clf.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        results.append({
            'max_depth': depth,
            'criterion': criterion,
            'val_accuracy': val_accuracy,
            'model': clf
        })
        
        print(f"\nDecision Tree (max_depth={depth}, criterion={criterion}):")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

# Find best model based on validation accuracy
best_result = max(results, key=lambda x: x['val_accuracy'])
best_clf = best_result['model']


# Evaluate best model on test set
y_test_pred = best_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")


# Function to predict price category for a new car
def predict_car_price_category(car_features):
    # Convert the input features to match the training data format
    car_df = pd.DataFrame([car_features])
    
    # Encode categorical variables
    for column in categorical_columns[:-1]:  # Exclude Price_Category
        if column in car_df.columns:
            car_df[column] = label_encoders[column].transform(car_df[column])
    
    # Make prediction
    prediction = best_clf.predict(car_df)[0]
    
    # Decode prediction
    return label_encoders['price_category'].inverse_transform([prediction])[0]

# Example usage
example_car = {
    'mileage': 50000,
    'registration_year': 2018,
    'previous_owners': 2,
    'fuel_type': 'Petrol',
    'body_type': 'Hatchback',
    'engine': 3.0,
    'gearbox': 'Manual',
    'doors': 5,
    'seats': 5,
    'service_history': True,
    'brand': 'BMW'
}
predicted_category = predict_car_price_category(example_car)
print(f"\nPredicted price category for the example car: {predicted_category}")


output_file = f'best_clf.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump(best_clf, f_out)

print(f'the model is saved to {output_file}')