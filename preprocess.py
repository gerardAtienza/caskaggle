import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    
    df = pd.read_csv(file_path)
    
    
    print("First few rows of the data:")
    print(df.head())
    
    for column in df.columns:
        if df[column].dtype == 'object':
            try:
                df[column] = df[column].str.replace('$', '').str.replace(',', '').str.replace(' ', '').astype(float)
            except ValueError:
                pass
    
    # Drop irrelevant columns
    columns_to_drop = ['LoanNr_ChkDgt', 'Name', 'ApprovalDate', 'ApprovalFY', 'DisbursementDate', 'ChgOffDate', 'City', 'Bank', 'ChgOffPrinGr', 'BalanceGross']  
    df.drop(columns=columns_to_drop, inplace=True)
    
    # Select columns to normalize and encode
    columns_to_normalize = ['NAICS', 'CreateJob', 'RetainedJob', 'DisbursementGross', 'GrAppv', 'SBA_Appv']  
    categorical_columns = ['UrbanRural', 'State', 'BankState']
    #binary_categorical_columns = ['NewExist', 'RevLineCr', 'LowDoc']  
    
    df['FranchiseCode'].fillna(0, inplace=True)
    df['FranchiseCode'] = df['FranchiseCode'].apply(lambda x: 1 if x == 1 else 2)
    
    # Map binary categorical columns to 1 and 0
    binary_mapping = {
        'NewExist': {1: 1, 2: 0},
        'RevLineCr': {'Y': 1, 'N': 0},
        'LowDoc': {'Y': 1, 'N': 0}
    }
    for col, mapping in binary_mapping.items():
        df[col] = df[col].map(mapping)
    
    # Encode categorical columns using one-hot encoding
    df = pd.get_dummies(df, columns=categorical_columns)

    # Drop columns with more than 50% missing values
    missing_values = df.isnull().mean()
    columns_to_drop = missing_values[missing_values > 0.5].index
    df.drop(columns=columns_to_drop, inplace=True)
    
    # Prepare the target variable
    df['MIS_Status'] = df['MIS_Status'].map({'P I F': 1, 'CHGOFF': 0})
    
    # Ensure no NaN values in the target variable
    df = df.dropna(subset=['MIS_Status'])
    
    y = df['MIS_Status']  # Target
    X = df.drop('MIS_Status', axis=1)  # Features

    # Check for highly correlated features
    correlation_matrix = X.corrwith(y).abs()
    print("Correlation with target variable:")
    print(correlation_matrix.sort_values(ascending=False))

    # Split the data into training and testing sets to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Impute missing values separately for training and testing sets
    for column in X_train.columns:
        if X_train[column].dtype == 'object':
            X_train[column] = X_train[column].fillna(X_train[column].mode()[0])
            X_test[column] = X_test[column].fillna(X_train[column].mode()[0])
        else:
            X_train[column] = X_train[column].fillna(X_train[column].median())
            X_test[column] = X_test[column].fillna(X_train[column].median())

    # Normalize columns
    scaler = StandardScaler()
    X_train[columns_to_normalize] = scaler.fit_transform(X_train[columns_to_normalize])
    X_test[columns_to_normalize] = scaler.transform(X_test[columns_to_normalize])

    return X_train, X_test, y_train, y_test, scaler


# X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("SBAnational.csv")
# print(X_train.head())
# print(y_train.head())