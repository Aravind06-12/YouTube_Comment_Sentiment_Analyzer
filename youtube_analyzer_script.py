import os
import googleapiclient.discovery
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import matplotlib.pyplot as plt

# Load API key from environment variable
# YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# For local testing, you can place your key directly here
YOUTUBE_API_KEY = "AIzaSyAFndyFbNdvrclRZNT1LTqbLOn-j9m97d4"

# Ensure necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

def fetch_comments(video_id, max_results=100):
    """Fetches YouTube comments for a given video ID."""
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results
    )
    response = request.execute()
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append({
            'author': comment['authorDisplayName'],
            'text': comment['textDisplay'],
            'like_count': comment['likeCount'],
            'published_at': comment['publishedAt']
        })
    return pd.DataFrame(comments)

def clean_text(text):
    """Performs text normalization on a string."""
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'https?:\/\/\S+', '', text) # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    return text

def analyze_sentiment(text):
    """Analyzes sentiment using VADER."""
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)

def main():
    video_id = "zSWdZVtXT7E"  # Example video ID (Interstellar)
    
    print("Fetching YouTube comments...")
    comments_df = fetch_comments(video_id, max_results=100)
    
    print("Cleaning and normalizing text...")
    comments_df['clean_text'] = comments_df['text'].apply(clean_text)
    
    print("Performing sentiment analysis...")
    comments_df['sentiment_scores'] = comments_df['clean_text'].apply(analyze_sentiment)
    
    comments_df['compound'] = comments_df['sentiment_scores'].apply(lambda score: score['compound'])
    
    def get_sentiment_label(score):
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'
    
    comments_df['sentiment_label'] = comments_df['compound'].apply(get_sentiment_label)
    
    # Save results to a CSV file
    output_path = "data/sentiment_analysis_results.csv"
    comments_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Plotting the results
    sentiment_counts = comments_df['sentiment_label'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['green', 'red', 'blue'])
    plt.title('Sentiment Distribution of YouTube Comments')
    plt.savefig('data/sentiment_pie_chart.png')
    print("Pie chart saved to data/sentiment_pie_chart.png")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    main()
