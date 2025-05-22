import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict

# Step 1: Load UCI Adult Dataset from URL
def load_adult_data():
    url_train = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    # Load train data
    train = pd.read_csv(url_train, header=None, names=column_names, na_values=' ?')
    # Load test data
    test = pd.read_csv(url_test, header=0, names=column_names, na_values=' ?')
    
    # Combine train and test for full dataset, we'll split later
    data = pd.concat([train, test], ignore_index=True)
    
    # Drop rows with missing values
    data = data.dropna()
    
    # Clean income labels
    data['income'] = data['income'].apply(lambda x: x.strip().strip('.'))
    
    return data

# Step 2: Preprocess Data
def preprocess_data(df):
    # Separate features and target
    X = df.drop('income', axis=1)
    y = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # We want to encode categorical variables with OneHotEncoder and scale numerical (optional)
    ct = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], remainder='passthrough')

    return X, y, ct

# Step 3: Train baseline model and check sex-based accuracy
def train_and_evaluate(X_train, X_test, y_train, y_test, ct, sample_weights=None):
    # Create a pipeline with preprocessing and logistic regression
    model = Pipeline([
        ('preprocessor', ct),
        ('clf', LogisticRegression(max_iter=500))
    ])

    # Train the model with or without sample weights
    model.fit(X_train, y_train, clf__sample_weight=sample_weights)

    # Predict on test set
    y_pred = model.predict(X_test)
 # Overall accuracy
    overall_acc = accuracy_score(y_test, y_pred)

    # Accuracy by sex group
    sex_test = X_test['sex'].values
    acc_by_sex = {}
    for sex in np.unique(sex_test):
        idx = sex_test == sex
        acc_by_sex[sex] = accuracy_score(y_test[idx], y_pred[idx])

    return model, overall_acc, acc_by_sex, y_pred

# Step 4: Calculate sample weights for reweighting to mitigate bias
def compute_reweighting_weights(X, y, protected_attr='sex'):
    df = X.copy()
    df['label'] = y

    # Calculate P(y)
    p_y = y.mean()

    # Calculate P(protected)
    p_protected = df[protected_attr].value_counts(normalize=True)

    # Calculate P(y | protected)
    p_y_given_protected = df.groupby(protected_attr)['label'].mean()

  # Calculate weights: w = P(y) * P(protected) / P(y, protected)
    # where P(y, protected) = P(y|protected)*P(protected)
    weights = []
    for i, row in df.iterrows():
        sex_val = row[protected_attr]
        label_val = row['label']
        p_ygp = p_y_given_protected[sex_val] if label_val == 1 else (1 - p_y_given_protected[sex_val])
        p_p = p_protected[sex_val]
        w = (p_y if label_val == 1 else (1 - p_y)) * p_p / (p_ygp * p_p)
        weights.append(w)
    return np.array(weights)

def main():
    print("Loading dataset...")
    data = load_adult_data()

    print("Preprocessing data...")
    X, y, ct = preprocess_data(data)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print("Training baseline model...")
    model, overall_acc, acc_by_sex, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test, ct)
    print(f"Baseline overall accuracy: {overall_acc:.4f}")
    print("Accuracy by sex:")
    for sex, acc in acc_by_sex.items():
        print(f"  {sex}: {acc:.4f}")

    print("\nComputing sample weights for reweighting bias mitigation...")
    sample_weights = compute_reweighting_weights(X_train, y_train, protected_attr='sex')

    print("Retraining model with reweighting...")
    model_rw, overall_acc_rw, acc_by_sex_rw, y_pred_rw = train_and_evaluate(X_train, X_test, y_train, y_test, ct, sample_weights=sample_weights)
    print(f"Reweighted model overall accuracy: {overall_acc_rw:.4f}")
    print("Accuracy by sex after reweighting:")
    for sex, acc in acc_by_sex_rw.items():
        print(f"  {sex}: {acc:.4f}")

if __name__ == "__main__":
    main()