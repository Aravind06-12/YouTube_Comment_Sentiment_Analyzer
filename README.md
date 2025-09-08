
# YouTube Comment Sentiment Analyzer and Predictor
This project demonstrates an end-to-end process for analyzing and predicting the sentiment of YouTube comments. It includes a complete data pipeline from scraping comments via the YouTube Data API to training a machine learning model.

## Project Features
**Data Scraping**: Fetches comments from a specified YouTube video using the YouTube Data API, including pagination to retrieve multiple pages of comments.

**Text Processing**: Cleans and normalizes the raw comment text by removing special characters, URLs, and stopwords.

**Sentiment Analysis (VADER)**: Uses the VADER lexicon to perform rule-based sentiment analysis on the comments, classifying them as Positive, Negative, or Neutral.

**Data Visualization**: Generates a bar chart, a pie chart, and a word cloud to visualize the sentiment distribution and most common words.

**Machine Learning Model**: Trains a Logistic Regression model using TF-IDF vectorization to predict sentiment on new, unseen comments.
