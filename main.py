import argparse
import os
from preprocess import load_and_preprocess_data
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from models import Model

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Train, tune, cross-validate, or test a model.')
    parser.add_argument('action', choices=['train', 'tune', 'cross_validate', 'test'], help='Action to perform')
    parser.add_argument('model', choices=['decision_tree', 'random_forest', 'gradient_boosting', 'xgboost', 'neural_network'], help='Model to use')
    args = parser.parse_args()

    # Load and preprocess the data
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("SBAnational.csv")

    # Print the target and features for verification
    print("Training target distribution:")
    print(y_train.value_counts())
    print("Test target distribution:")
    print(y_test.value_counts())
    print("First few rows of X_train:")
    print(X_train.head())
    print("First few rows of X_test:")
    print(X_test.head())

    # Define parameter grids
    dt_param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10]
    }
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 30, 50],
        'min_samples_split': [2, 5, 10]
    }
    gb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }
    
    # Initialize the model
    if args.model == 'decision_tree':
        model = Model(DecisionTreeClassifier(class_weight='balanced'), dt_param_grid)
    elif args.model == 'random_forest':
        model = Model(RandomForestClassifier(class_weight='balanced'), rf_param_grid)
    elif args.model == 'gradient_boosting':
        model = Model(GradientBoostingClassifier(), gb_param_grid)
    elif args.model == 'xgboost':
        model = Model(xgb.XGBClassifier(), xgb_param_grid)
    
    # Perform the specified action
    if args.action == 'train':
        model.train(X_train, y_train)
        model.save_model(f"{args.model}_model.h5" if args.model == 'neural_network' else f"{args.model}_model.pkl")
    elif args.action == 'tune':
        model.tune_hyperparameters(X_train, y_train)
        model.save_model(f"{args.model}_model.pkl")
        print("Best parameters found: ", model.get_best_params())
    elif args.action == 'cross_validate': 
        model_path = f"{args.model}_model.pkl"
        Model.load_model(model_path)
        if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist. Please train the model first.")
        k_values = [5, 10]
        cv_results = model.cross_validate(X_train, y_train, k_values)
        print(f"Cross-Validation Results: {cv_results}")
    elif args.action == 'test':
        model_path = f"{args.model}_model.pkl"
        Model.load_model(model_path)
        if not os.path.exists(model_path):
            print(f"Model file {model_path} does not exist. Please train the model first.")
        test_results = model.test(X_test, y_test)
        print(f"Test Results: {test_results}")

if __name__ == "__main__":
    main()