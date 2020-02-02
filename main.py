from flask import Flask
from flask import render_template
from flask import request

#ml package
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("home.html")

@app.route('/',methods=['POST'])
def predict():
    df = pd.read_csv("youtube_spam.csv",header=0)
    feature = df['CONTENT']
    y = df.CLASS
    cv = CountVectorizer()
    X = cv.fit_transform(feature)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    if request.method == "POST":
        comment = request.form['message']
        x_pred = [comment]
        transform = cv.transform(x_pred)
        y_pred = model.predict(transform)
        return render_template("response.html", prediction = y_pred)
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)