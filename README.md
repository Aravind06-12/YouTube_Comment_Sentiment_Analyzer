# YouTube Comment Sentiment Analyzer

This project builds a data pipeline to scrape comments from a specific YouTube video, perform sentiment analysis on them, and visualize the results. The entire workflow is run within a Jupyter Notebook.

## Project Steps

1.  **Scraping Comments**: Fetches YouTube comments for a given video using the YouTube Data API.
2.  **Data Processing**: The comments are converted into a structured Pandas DataFrame for easier manipulation.
3.  **Sentiment Analysis**: The VADER sentiment lexicon is used to classify each comment as positive, negative, or neutral.
4.  **Visualization**: Pie and bar charts are generated to show the distribution of sentiment.

## How to Run

1.  Clone this repository.
2.  Set up your YouTube Data API key as an environment variable (see the note below).
3.  Install all the required Python libraries using `pip install -r requirements.txt`.
4.  Open and run the `Youtube_Comment_Sentiment_Analyzer.ipynb` notebook.
