from flask import Flask, jsonify,request, render_template
import json
import firebase_admin
from firebase_admin import credentials, db

import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from flask_cors import CORS

cred = credentials.Certificate("analyzesentiment-41c85-firebase-adminsdk-dl7tz-a2996ff573.json")
firebase_admin = firebase_admin.initialize_app(cred, {'databaseURL': 'https://analyzesentiment-41c85-default-rtdb.asia-southeast1.firebasedatabase.app'})
from sklearn.metrics.pairwise import cosine_similarity

# from surprise import Dataset, Reader, SVD

ref = db.reference("/")
ref = db.reference("/ReviewAndRating/")
data = ref.get()



from sqlalchemy import create_engine
import requests
import joblib 

# Importing needed libraries
import numpy as np
from ast import literal_eval #module that converts a string of lists to a normal list
# hotelData = pd.read_csv('Hotel_reviews.csv')
# hotelData.head()

# print(hotelData)

# print(hotelData.head())

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

# print(rating_counts , "hello111")
# print(sentiment_counts , "hello155")

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

@app.route('/api/hotelsList', methods=['GET'])
def send_hotelsList():
        return jsonify('hello json skjdjfjsjsjkjfjsj')
        # return jsonify({"accuracy": accuracy})






# Sample hotel data
# hotel_data = pd.DataFrame([
#     {"name": "Hotel A", "location": "Hyderabad", "rating": 4.5},
#     {"name": "Hotel B", "location": "Hyderabad", "rating": 4.0},
#     {"name": "Hotel C", "location": "Bangalore", "rating": 4.2},
#     {"name": "Hotel D", "location": "Mumbai", "rating": 4.8},
#     # Add more hotel data as needed
# ])

# Read the CSV file
file_path = 'Hotel_reviews.csv'
hotel_data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame



# Read the CSV file into a DataFrame
hotel_data = pd.read_csv('Hotel_reviews.csv', header=None, names=[
    "Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Hotel_Name", 
    "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", 
    "Total_Number_of_Reviews", "Positive_Review", "Review_Total_Positive_Word_Counts", 
    "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", 
    "days_since_review", "lat", "lng"
])

# Select relevant columns and rename them for convenience
hotel_data = hotel_data[['Hotel_Name', 'Hotel_Address', 'Reviewer_Score']]
hotel_data.columns = ['name', 'location', 'rating']

# Drop rows with missing values in 'location' or 'name'
hotel_data.dropna(subset=['location', 'name'], inplace=True)

# Convert 'location' and 'name' columns to string type to avoid type conflicts
hotel_data['location'] = hotel_data['location'].astype(str)
hotel_data['name'] = hotel_data['name'].astype(str)

# Combine relevant features into a single string for each hotel
hotel_data['features'] = hotel_data['location'] + ' ' + hotel_data['name']

print(hotel_data)

# Vectorize the combined features using TF-IDF
vectorizer = TfidfVectorizer()
feature_matrix = vectorizer.fit_transform(hotel_data['features'])


def get_recommendations(input_text, hotel_data, feature_matrix, n=5):
    input_vec = vectorizer.transform([input_text])
    similarities = cosine_similarity(input_vec, feature_matrix).flatten()
    similar_indices = similarities.argsort()[-n:][::-1]
    recommendations = hotel_data.iloc[similar_indices]
    return recommendations[['name', 'location', 'rating']]


# # Convert the DataFrame to the Surprise dataset
# reader = Reader(rating_scale=(1, 5))
# data = Dataset.load_from_df(hotel_data[['user_id', 'hotel_name', 'rating']], reader)



# # Train-test split
# trainset, testset = train_test_split(data, test_size=0.25)

# # Use the SVD algorithm for matrix factorization
# algo = SVD()
# # Train the algorithm on the trainset
# algo.fit(trainset)

# # Test the algorithm on the testset
# predictions = algo.test(testset)
# accuracy.rmse(predictions)

# def get_recommendations(user_id, algo, hotel_list, n=5):
#     all_hotels = hotel_list['hotel_name'].unique()
#     rated_hotels = hotel_data[hotel_data['user_id'] == user_id]['hotel_name'].values
#     unrated_hotels = [hotel for hotel in all_hotels if hotel not in rated_hotels]
#     predictions = [algo.predict(user_id, hotel) for hotel in unrated_hotels]
#     recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)
#     top_recommendations = recommendations[:n]
#     return [(rec.iid, rec.est) for rec in top_recommendations]


@app.route('/api/recommend_hotels', methods=['POST'])
def recommend_hotels():
    request_data = request.json
    input_text = request_data.get('inputText', '').lower()
    print('user_location')
    if input_text:
        recommendations = get_recommendations(input_text, hotel_data, feature_matrix)
        if not recommendations.empty:
            return recommendations.to_json(orient='records')
        else:
            return jsonify({"message": f"No hotels found matching '{input_text}'"}), 404
    else:
        return jsonify({"message": "Input text not provided in request"}), 400
    


@app.route('/api/train', methods=['POST'])
def train():
    token = request.headers.get('x-api-token')
    if not token:
        return jsonify({'error': 'Authorization token is missing'}), 400
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    status = data.get('status', 'tripall')
    fetched_data = fetch_data(token, status)
    if fetched_data:
        print("Data fetched successfully:")
        train_model(fetched_data)
        global model
        model = joblib.load('model.pkl')
        return jsonify({'message': 'Model trained successfully'})
    else:
        return jsonify({'error': 'Failed to fetch data'}), 400

# Database connection details
DB_HOST = "localhost"
DB_USER = "avplat"
DB_PASS = "g=gP32?TewVEdAtS"
DB_NAME = "avplat"
SERVER_CONNECTION_STRING = "mysql+pymysql://avplat:g=gP32?TewVEdAtS@avplat-staging-upgraded-cluster.cluster-c8agfa7vorgz.ca-central-1.rds.amazonaws.com/avplat"
engine = create_engine(SERVER_CONNECTION_STRING)




def fetch_data(token, status):
    url = "https://sapi.avplat.com/public/index.php?page=API&action=GetQuoteAll"
    headers = {
        "X-Api-Token": token,
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    payload = {
        "enct": 1,
        "status": status
    }
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        return response.json()  # Assuming the API returns JSON
    else:
        return None

# Function to train the model
def train_model(data):
    # Assuming the data is in a format that can be converted to a DataFrame
    df = pd.DataFrame(data['GetQuoteAll'])
    
    # Example feature extraction
    X = df[['SRID', 'AircraftType', 'depart']]
    y = df['Status']
    
    # Convert categorical data to numeric (this is just an example)
    X = pd.get_dummies(X)
    
    # Initialize and train the model (RandomForestClassifier in this case)
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Save the trained model to a file
    joblib.dump(model, 'model.pkl')


@app.route('/api/search', methods=['POST'])
def search():
    token = request.headers.get('x-api-token')
    if not token:
        return jsonify({'error': 'Authorization token is missing'}), 400
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    query = data.get('query')
    print(query , "queryquery")
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    result = model.predict(query)
    print(result , "resultresultresult")
if __name__ == "__main__":
    app.run(debug=DEVELOPMENT_ENV)