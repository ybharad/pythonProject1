# spam_classifier.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# Sample data
data = [
    ("Buy cheap luxury watches!!!", 1),
    ("Hello, how are you?", 0),
    # Add more data here
]

# Preprocess the data
texts, labels = zip(*data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X, y)

# Function to predict whether a text is spam or not
def predict_spam(text):
    text_vectorized = vectorizer.transform([text])
    prediction = classifier.predict(text_vectorized)
    return prediction[0]  # 1 for spam, 0 for not spam

