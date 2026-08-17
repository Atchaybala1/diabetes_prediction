import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_model(file_path="/content/Diabetes_prediction.csv", model_filename="diabetes_prediction_model.joblib"):
    """Trains a diabetes prediction model and saves it."""

    # Load the dataset
    df = pd.read_csv(file_path)

    # Separate features (X) and target (y)
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the RandomForestClassifier model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")

if __name__ == '__main__':
    train_model()
