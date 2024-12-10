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

# Save the trained model to a file
joblib.dump(model, 'breast_cancer_model.pkl')

# Render the index.html template
@app.route('/')
def index():
    return render_template('index1.html')

# Endpoint for receiving data and making predictions
# Endpoint for receiving data and training the model
@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        print("Received data for training:", data)
        return "Data received for training successfully!"
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()

        # Ensure that the input_data contains all the required features
        required_features = X.columns.tolist()
        for feature in required_features:
            if feature not in input_data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        # Convert input data to the correct data types (all numeric)
        for feature in required_features:
            input_data[feature] = float(input_data[feature])

        # Convert input data to a numpy array
        input_data_as_list = [input_data[feature] for feature in required_features]
        input_data_as_numpy_array = np.asarray(input_data_as_list)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Load the trained model from the file
        loaded_model = joblib.load('breast_cancer_model.pkl')

        # Make predictions
        prediction = loaded_model.predict(input_data_reshaped)

        if prediction[0] == 0:
            result = "The Breast cancer is Malignant"
        else:
            result = "The Breast Cancer is Benign"

        return jsonify({"prediction": result})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


#accuracy on test data
# X_test_prediction = model.predict(X_test)
# test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

# print('Accuracy on test data = ', test_data_accuracy)

# """Building a Predictive System"""

# input_data = (17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
# )

# # change the input data to a numpy array
# input_data_as_numpy_array = np.asarray(input_data)

# # reshape the numpy array as we are predicting for one datapoint
# input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# prediction = model.predict(input_data_reshaped)
# print(prediction)

# if (prediction[0] == 0):
#   print('The Breast cancer is Malignant')

# else:
#   print('The Breast Cancer is Benign')
