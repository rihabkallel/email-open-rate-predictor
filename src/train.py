# train.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib

class ModelTrainer:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=2000)
        self.model = RandomForestRegressor()

    def train_model(self, data):
        # Define the preprocessor for the text data
        text_transformer = Pipeline(steps=[
            ('tfidf', self.tfidf)
        ])

        # Define the preprocessor for the numerical features
        numeric_features = ['length', 'digit_count', 'exclamation_count']
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Combine the text and numeric preprocessors
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', text_transformer, 'clean_subject'),
                ('num', numeric_transformer, numeric_features)
            ]
        )

        # Create the full pipeline with the preprocessor and the model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', self.model)
        ])

        y = data['open_rate']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0)

        # Hyperparameter tuning using Grid Search
        param_grid = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        }

        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_

        # Predicting open rates
        y_pred = best_pipeline.predict(X_test)

        # Evaluating the model
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print(f'Best Parameters: {grid_search.best_params_}')
        print(f'RMSE: {rmse}')
        print(f'R^2 Score: {r2}')
        print(f'MAPE: {mape * 100:.2f}%')

        # Save the trained pipeline
        joblib.dump(best_pipeline, 'model/open_rate_predictor.pkl')

        return best_pipeline, mse, rmse, r2, mape

    def cross_validate_model(self, data):
        # Define the preprocessor for the text data
        text_transformer = Pipeline(steps=[
            ('tfidf', self.tfidf)
        ])

        # Define the preprocessor for the numerical features
        numeric_features = ['length', 'digit_count', 'exclamation_count']
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # Combine the text and numeric preprocessors
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', text_transformer, 'clean_subject'),
                ('num', numeric_transformer, numeric_features)
            ]
        )

        # Create the full pipeline with the preprocessor and the model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', self.model)
        ])

        y = data['open_rate']

        # Cross-validation
        scores = cross_val_score(pipeline, data, y, cv=5, scoring='r2')
        print(f'Cross-Validation R^2 Scores: {scores}')
        print(f'Average Cross-Validation R^2 Score: {scores.mean()}')
        return scores
