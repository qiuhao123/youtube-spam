from flask import Flask
from flask import render_template
from flask import request

#ml package
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('home.html')
# # @app.route('/table')
# # def display():
# if __name__ == 'main':
#     app.run(host="127.0.0.1",port=8080,debug=True)
@app.route('/')
def hello():
    return render_template("home.html")

@app.route('/',methods=['POST'])
def predict():
    url = "https://github.com/qiuhao123/youtube-spam/blob/master/youtube_spam.csv"
    df = pd.read_csv(url)
    df_data = df[["CONTENT", "CLASS"]]
    # Features and Labels
    df_x = df_data['CONTENT']
    df_y = df_data.CLASS
    # Extract Feature With CountVectorizer
    corpus = df_x
    cv = CountVectorizer()
    X = cv.fit_transform(corpus) # Fit the Data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)
    #Naive Bayes Classifier
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    #Alternative Usage of Saved Model
    # ytb_model = open("naivebayes_spam_model.pkl","rb")
    # clf = joblib.load(ytb_model)
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        return render_template("response.html", prediction = my_prediction, comment = comment)
    #return render_template('response.html',prediction=prediction,)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)