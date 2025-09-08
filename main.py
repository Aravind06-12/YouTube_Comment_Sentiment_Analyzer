from googleapiclient.discovery import build
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

API_KEY = "AIzaSyAYGuAsTiN5iNlBuQHH4rjac3QzjqyNUdI"
VIDEO_ID = "ZT2ilX9MC1w"  # Example video ID

youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_comments(videoId, max_pages=2):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=videoId,
        maxResults=100,
        textFormat="plainText"
    )
    page = 0
    while request and page < max_pages:
        response = request.execute()
        for item in response['items']:
            snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'author': snippet['authorDisplayName'],
                'text': snippet['textDisplay'],
                'published_at': snippet['publishedAt']
            })
        request = youtube.commentThreads().list_next(request, response)
        page += 1
    return comments

print("Fetching YouTube comments...")
comments = get_comments(VIDEO_ID, max_pages=2)
print(f"Fetched {len(comments)} comments.")

df = pd.DataFrame(comments)

# Text cleaning and stopword removal
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(clean_text)

# Sentiment scores
analyzer = SentimentIntensityAnalyzer()
def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['clean_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['sentiment_label'] = df['sentiment'].apply(classify_sentiment)

print(df[['text', 'clean_text', 'sentiment', 'sentiment_label']].head())

# Visualization
sentiment_counts = df['sentiment_label'].value_counts()
plt.figure(figsize=(6,4))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,5))
sentiment_counts.plot(kind='pie', autopct='%1.1f%%', colors=['green','red','gray'])
plt.title("Sentiment Percentage")
plt.ylabel("")
plt.tight_layout()
plt.show()

all_text = " ".join(df['clean_text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Common Words")
plt.tight_layout()
plt.show()

# ML Model
X = df['clean_text']
y = df['sentiment_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Predict new comments
new_comments = ["This video is amazing!", "I didn’t like the content", "It was okay, not too great"]
new_comments_tfidf = vectorizer.transform(new_comments)
predictions = model.predict(new_comments_tfidf)
for comment, label in zip(new_comments, predictions):
    print(f"{comment} → {label}")

# Load and predict (example)
loaded_model = joblib.load("sentiment_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")
test_comments = ["The editing in this video is fantastic!", "Worst video I’ve ever seen", "It’s fine, nothing special"]
test_tfidf = loaded_vectorizer.transform(test_comments)
predictions = loaded_model.predict(test_tfidf)
for comment, sentiment in zip(test_comments, predictions):
    print(f"{comment} → {sentiment}")
