import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, request, jsonify

df = pd.read_csv("emails_dataset.csv")
print(df.head())
print(df.shape)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["email"])
y = df["label"]
model = MultinomialNB()
model.fit(X, y)
app = Flask(__name__)
@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["email"]
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return jsonify({"classe": prediction[0]})
app.run(host="0.0.0.0", port=5000)
