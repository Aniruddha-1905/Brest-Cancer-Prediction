from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import warnings

app = Flask(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

# Data Collection & Processing
data_frame = pd.read_csv('dataset.csv')
data_frame.rename(columns={'diagnosis': 'label'}, inplace=True)
data_frame['label'] = data_frame['label'].map({'M': 0, 'B': 1})

# Separating the features and target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

# Splitting the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Model Training: Logistic Regression
model = LogisticRegression()
model.fit(X_train, Y_train)

@app.route('/')
def index():
    return render_template('trial.html')

@app.route('/train', methods=['POST'])
def train_and_predict():
    try:
        input_data = request.get_json()

        # Ensure that the input_data contains all the required features
        required_features = X.columns.tolist()
        missing_features = [feature for feature in required_features if feature not in input_data]
        if missing_features:
            return jsonify({"error": f"Missing features: {', '.join(missing_features)}"}), 400

        # Create a DataFrame from the input data
        input_df = pd.DataFrame(input_data, index=[0])

        # Convert the input DataFrame to have the same data types as the original dataset
        input_df = input_df.astype(X.dtypes)

        # Fill any missing values with a default value (e.g., 0)
        input_df = input_df.fillna(0)

        # Make predictions
        prediction = model.predict(input_df)
        if prediction[0] == 0:
            result = "The Breast cancer is Malignant"
        else:
            result = "The Breast cancer is Benign"
        return jsonify({"prediction": result})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)