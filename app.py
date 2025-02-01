from flask import Flask, request, jsonify
import pickle

with open('modelDCT.pkl', 'rb') as model_file:
    modelDCT = pickle.load(model_file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorization = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    input_text = [data['text']]
    input_data = vectorization.transform(input_text)
    prediction = modelDCT.predict(input_data)


    result = 'Real news' if prediction[0] == 1 else 'Fake news'
    return jsonify({'prediction': result})



if __name__ == '__main__':
    app.run(debug=True)

