#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train forecasting model.

Author:
    Erik Johannes Husom

Created:
    2024-03-21

"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from config import config

# Function to preprocess the data
def preprocess_data(df, target_column, scale_features=False):

    # Exclude non-numeric values
    # df = df.select_dtypes(exclude=["object", "category"])
    # df = df.dropna()

    # # Applying One-Hot Encoding to non-numeric columns
    # df = pd.get_dummies(df, columns=non_numeric_columns)

    unique_val_threshold = 10

    columns_to_drop = [
            # target_column,
            "energy_consumption_monitoring",
            "total_duration",
            "response_duration",
            "response_token_length",
            "prompt_duration",
            "index",
            "load_duration",
            "Unnamed: 0",
    ]

    df = df.drop(columns=columns_to_drop)
    # y = df[target_column]

    # Drop target column and rows with NA in target column
    df = df.dropna(subset=[target_column])
    y = df[target_column]
    df = df.drop(columns=[target_column])

    # Identify numeric, text, and categorical columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    potential_categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()

    text_columns = [col for col in potential_categorical if df[col].nunique() > unique_val_threshold]
    categorical_columns = list(set(potential_categorical) - set(text_columns))


    # Define transformers
    transformers = []
    if scale_features:
        transformers.append(('num', StandardScaler(), numeric_columns))
    if text_columns:
        transformers.append(('text', TfidfVectorizer(), text_columns))
    if categorical_columns:
        transformers.append(('cat', OneHotEncoder(), categorical_columns))

    # Column Transformer
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

    # Applying the ColumnTransformer
    # Note: This step now includes fitting the transformer, so it should be applied here rather than in the model pipeline.
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test

    # # Splitting the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # if scale_features:
    #     scaler = StandardScaler()
    #     X_train = scaler.fit_transform(X_train)
    #     X_test = scaler.transform(X_test)

    # return X_train, X_test, y_train, y_test

def neural_network(input_shape=57):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_shape,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(), loss='mse')

    return model

# Function to select the machine learning model
def select_model(model_name):
    models = {
        "random_forest": RandomForestRegressor(),
        "decision_tree": DecisionTreeRegressor(),
        "gradient_boosting": GradientBoostingRegressor(),
        "xgboost": xgb.XGBRegressor(),
        "svm": SVR(),
        "linear_regression": LinearRegression(),
        "neural_network": neural_network(),
    }
    return models.get(model_name, LinearRegression())  # Default to Linear Regression

# Function to train the model and evaluate it
def train_and_evaluate(model_name, X_train, X_test, y_train, y_test):
    model = select_model(model_name)

    if model_name == "neural_network":
        model.fit(X_train, y_train, epochs=1000)
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model: {model_name}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R^2: {r2}")

    if r2 > 0.5:
        plot_true_vs_predicted(y_test, y_pred, model_name)

    return model, mse, mae, r2

def plot_true_vs_predicted(y_true, y_pred, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', label='Predicted')
    plt.plot(y_true, y_true, color='red', label='True')  # Line for perfect predictions
    plt.title(f'True vs. Predicted Energy Consumption - {model_name}')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def new_train(df):
    # Preparing the dataset
    X = df[['prompt', 'type']]
    y = df['energy_consumption_llm']

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Defining the transformation for text and categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(), 'prompt'),
            ('cat', OneHotEncoder(), ['type'])
        ])

    # Creating a modeling pipeline
    model = make_pipeline(preprocessor, RandomForestRegressor(n_estimators=100))

    # Training the model
    model.fit(X_train, y_train)

    # Predicting and evaluating
    y_pred = model.predict(X_test)
    print("RMSE: ", mean_squared_error(y_test, y_pred, squared=False))
    print("R2: ", r2_score(y_test, y_pred))

    plot_true_vs_predicted(y_test, y_pred, "RandomForestRegressor")


# Example usage
if __name__ == "__main__":
    df = pd.read_csv(config.MAIN_DATASET_WITH_FEATURES_PATH)  # Replace with your dataset path
    X_train, X_test, y_train, y_test = preprocess_data(df, "energy_consumption_llm", scale_features=True)

    # Easily switch between models by changing the model name
    model_names = ["random_forest", "decision_tree", "gradient_boosting", "xgboost", "svm", "linear_regression"]#, "neural_network"]
    # model_names = ["linear_regression"]
    for model_name in model_names:
        train_and_evaluate(model_name, X_train, X_test, y_train, y_test)

    # new_train(df)
