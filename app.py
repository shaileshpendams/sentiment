from flask import Flask, jsonify, render_template
import json
import firebase_admin
from firebase_admin import credentials, db



from flask import Flask, render_template, request
import json
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask_cors import CORS

cred = credentials.Certificate("analyzesentiment-41c85-firebase-adminsdk-dl7tz-1a72dc3076.json")
firebase_admin = firebase_admin.initialize_app(cred, {'databaseURL': 'https://analyzesentiment-41c85-default-rtdb.asia-southeast1.firebasedatabase.app'})

ref = db.reference("/")
ref = db.reference("/ReviewAndRating/")
data = ref.get()


app = Flask(__name__)
CORS(app)


# Initialize empty lists to store comments and ratings
comments = []
ratings = []
reviewList = []
rating_comment_df:any

df = pd.DataFrame(data)
# Iterate through the JSON data to extract comments and ratings
for key, value in df.items():
    for user_id, reviews in data.items():
     for index, review_list in reviews.items():
        for review in review_list:
                reviewList.append({
                'user_id': user_id,
                'name' :review.get('name', ''),
                'comment': review.get('comment', ''),
                'rating': review.get('rating', 0),
                'rating_stars': '★' * review.get('rating', 0),
                'image':review.get('image', '')
                })
                if 'comment' in review:
                   comments.append(review['comment'])

                if 'rating' in review:
                  ratings.append(review['rating'])
                #   print( len(comments) , '//2627') 
                #   print( len(ratings) ,'//2479')  
                
# Check lengths and trim the longer list if they are unequal
if len(ratings) != len(comments):
    min_length = min(len(ratings), len(comments))
    ratings = ratings[:min_length]
    comments = comments[:min_length]

# Check if lengths are equal
# if len(ratings) == len(comments):
    # Creating the DataFrame
rating_comment_df = pd.DataFrame({
        'rating': ratings,
        'comment': comments
    }) 

DEVELOPMENT_ENV = True


# Data Preprocessing
# Assuming comment preprocessing
stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

# Function to preprocess comments
def preprocess_comment(comment):
    tokens = word_tokenize(comment.lower())
    processed_tokens = [porter.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(processed_tokens)

# if data:
#     comments = []
#     ratings = []
#     for user_id, reviews_data in data.items():
#         for review in reviews_data['reviews']:
#             comment = review.get('comment', '')
#             rating = review.get('rating', 0)
#             comments.append(comment)
#             ratings.append(rating)

rating_comment_df['processed_comment'] = rating_comment_df['comment'].apply(preprocess_comment)
rating_counts = rating_comment_df['rating'].value_counts()

# print(rating_comment_df)

 # Example Modeling - Sentiment analysis
def sentiment_analysis(rating):
        if rating >= 4:
            return 'Positive'
        elif rating <= 2:
            return 'Negative'
        else:
            return 'Neutral'

rating_comment_df['sentiment'] = rating_comment_df['rating'].apply(sentiment_analysis)

sentiment_counts = rating_comment_df['sentiment'].value_counts()

print(rating_counts , "hello111")
print(sentiment_counts , "hello155")

# Your actual model training, evaluation, and predictions can go here
 # For demonstration, initializing a basic RandomForestClassifier
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(rating_comment_df['comment'])
y = rating_comment_df['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred, output_dict=True)
classification_rep = classification_report(y_test, y_pred, output_dict=True, zero_division=1)

print(sentiment_counts , "hello")

@app.route('/api/feedback', methods=['GET'])
def send_feedback():
    if reviewList:
        return jsonify(reviewList)
    else:
        return jsonify({"message": "No sentiment data available"})

@app.route('/api/send_sentiment_results', methods=['GET'])
def send_sentiment_results():
    if not sentiment_counts.empty:
        # Convert the Series to a dictionary with keys as indices and values as counts
        sentiment_dict = sentiment_counts.to_dict()
        return jsonify(sentiment_dict)
    else:
        return jsonify({"message": "No sentiment data available"})

@app.route('/api/send_rating_results', methods=['GET'])
def send_rating_results():
     if not rating_counts.empty:
        rating_dict = rating_counts.to_dict()
        return jsonify(rating_dict)
     else:
        return jsonify({"message": "No rating data available"})

@app.route('/api/send_accuracy_results', methods=['GET'])
def send_accuracy_results():
     if not accuracy.empty:
        accuracy_dict = rating_counts.to_dict()
        return jsonify(accuracy_dict)
        # return jsonify({"accuracy": accuracy})
     else:
        return jsonify({"message": "No accuracy data available"})

if __name__ == "__main__":
    app.run(debug=DEVELOPMENT_ENV)