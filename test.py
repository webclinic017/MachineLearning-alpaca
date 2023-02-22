import time
from finvizfinance.quote import finvizfinance as fvf
import nltk
# nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas

with open('tickers.txt', 'r') as f:
    list_of_tickers = f.readlines()

sentiments = {}

list_of_tickers = [i.strip() for i in list_of_tickers]

for ticker in list_of_tickers:
    if '.' in ticker:
        ticker = ticker.replace('.', '-')

    stock = fvf(ticker)

    news = stock.ticker_news()

    vader = SentimentIntensityAnalyzer()

    vader_scores = news['Title'].apply(vader.polarity_scores).tolist()

    compound_scores = [i['compound'] for i in vader_scores]
    average_score = sum(compound_scores) / len(compound_scores)

    sentiments[ticker] = average_score

    print(f"{ticker}: {average_score}")

    time.sleep(3)


average_sentiment = sum([i for i in sentiments.values()]) / len(sentiments)
adjusted_sentiment = {}

for k, v in sentiments.items():
    adjusted_sentiment[k] = v-average_sentiment

print(sentiments)
print()
print(adjusted_sentiment)

