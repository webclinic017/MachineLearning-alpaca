from finvizfinance.quote import finvizfinance as fvf
import nltk
# nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas

ticker = 'AAPL'
stock = fvf(ticker)

news = stock.ticker_news()

vader = SentimentIntensityAnalyzer()

vader_scores = news['Title'].apply(vader.polarity_scores).tolist()


compound_scores = [i['compound'] for i in vader_scores]
print(compound_scores)




