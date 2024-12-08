import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import pickle


class Model:
    def __init__(self, model, param_grid):
        self.model = model
        self.param_grid = param_grid
        self.grid_search = None
        self.best_model = None
        self.best_params_ = None

    def tune_hyperparameters(self, X, y):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.grid_search = GridSearchCV(self.model, self.param_grid, cv=skf, scoring='f1', verbose=10)
        self.grid_search.fit(X, y)
        self.best_model = self.grid_search.best_estimator_
        self.best_params_ = self.grid_search.best_params_

    def train(self, X, y):
        if self.best_model is None:
            self.tune_hyperparameters(X, y)
        self.best_model.fit(X, y)

    def predict(self, X_test):
        return self.best_model.predict(X_test)

    def cross_validate(self, X, y, k_values):
        results = {}
        for k in k_values:
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            accuracies = []
            for train_index, test_index in skf.split(X, y):
                print(f"Training fold with {len(train_index)} samples and testing fold with {len(test_index)} samples.")
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                self.train(X_train, y_train)
                predictions = self.predict(X_test)
                fold_f1_score = f1_score(y_test, predictions)
                print(f"Fold F1 score: {fold_f1_score}")
                accuracies.append(fold_f1_score)
            results[k] = np.mean(accuracies)
        return results

    def test(self, X_test, y_test):
        predictions = self.predict(X_test)
        f1 = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': conf_matrix
        }

    def save_model(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)
            
    def get_best_params(self):
        return self.best_params_

