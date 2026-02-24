# predict.py
import argparse
import joblib
import numpy as np

# Load the trained model
model = joblib.load('model.joblib')

# Iris target names
target_names = ['setosa', 'versicolor', 'virginica']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict iris species from measurements.')
    parser.add_argument('--sepal_length', type=float, required=True, help='Sepal length (cm)')
    parser.add_argument('--sepal_width', type=float, required=True, help='Sepal width (cm)')
    parser.add_argument('--petal_length', type=float, required=True, help='Petal length (cm)')
    parser.add_argument('--petal_width', type=float, required=True, help='Petal width (cm)')
    args = parser.parse_args()

    features = np.array([[args.sepal_length, args.sepal_width,
                          args.petal_length, args.petal_width]])
    pred_class = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]

    print(f"Predicted species: {target_names[pred_class]}")
    print(f"Probabilities: setosa={pred_proba[0]:.3f}, versicolor={pred_proba[1]:.3f}, virginica={pred_proba[2]:.3f}")