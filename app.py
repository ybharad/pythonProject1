# app.py
from flask import Flask, request, render_template
from spam_classifier import predict_spam

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_text = request.form['feedback']
    # Store the feedback in a file, database, or any other storage method
    with open('feedback.txt', 'a') as feedback_file:
        feedback_file.write(feedback_text + '\n')
    return render_template('index.html', feedback_submitted=True)



@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = predict_spam(text)
    return render_template('index.html', original_text=text, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

