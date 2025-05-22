from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def main():
    # Load Iris dataset
    data = load_iris()
    X, y = data.data, data.target

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Initialize and train logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Predict and calculate accuracy on test set
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    # Save model to file
    joblib.dump(model, "model.pkl")
    print("Model saved to model.pkl")

if __name__ == "__main__":
    main()