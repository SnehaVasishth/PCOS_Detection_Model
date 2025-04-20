from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open("pcos_stacking_svm_lr.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Extract features in the correct order
        features = [
            data["Age"],
            data["Weight"],
            data["Height"],
            data["Blood Group"],
            data["Menstrual Cycle Interval"],
            data["Recent Weight Gain"],
            data["Skin Darkening"],
            data["Hair Loss"],
            data["Acne"],
            data["Regular Fast Food Consumption"],
            data["Regular Exercise"],
            data["Mood Swings"],
            data["Regular Periods"],
            data["Excessive Body/Facial Hair"],
            data["Menstrual Duration (Days)"]
        ]

        # Convert to numpy array and reshape for prediction
        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        return jsonify({'pcos_diagnosis': int(prediction[0])})

    except KeyError as e:
        return jsonify({'error': f'Missing field in input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
