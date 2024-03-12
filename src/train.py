#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train ML models on LLMEC data set.

Author:
    Erik Johannes Husom

Created:
    2024-03-08

"""
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from config import config

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def plot_true_vs_pred(y_true, y_pred):

    plt.figure(figsize=(10, 10))
    plt.scatter(y_true, y_pred)
    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.show()
    # plt.savefig(config.PLOTS_PATH / "true_vs_pred.png")


def simple_models():
    data = pd.read_csv(config.MAIN_DATASET_PATH)
    # Preprocess the prompts
    data['processed_prompt'] = data['prompt'].apply(preprocess_text)

    # Define features and target variable
    X = data['processed_prompt']
    y = data['energy_consumption_llm']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model_pipeline = Pipeline([
        ('vect', TfidfVectorizer()),
        ('impute', SimpleImputer(strategy='mean')),
        # ('clf', RandomForestRegressor(random_state=42))
        # ('clf', xgb.XGBRegressor())
    ])

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)

    plot_true_vs_pred(y_test, y_pred)
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R2 score: {r2_score(y_test, y_pred)}")


if __name__ == '__main__':
    simple_models()
