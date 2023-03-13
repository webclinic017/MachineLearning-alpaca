import time
from finvizfinance.quote import finvizfinance as fvf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup


def get_article(url, title):
    headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36'
    }
    try:
        get = requests.get(url=url, headers=headers).text
        parser = BeautifulSoup(get, 'html.parser')
        article = parser.find('article')
        ps = article.find_all('p')
        texts = []
        for i in ps:
            texts.append(i.get_text())

        line = (' '.join(texts)).strip()
    except:
        print("EXCEPTION")
        return analyze_article(title)

    # print(analyze_article(line))
    return analyze_article(line)


def analyze_article(text):
    vader = SentimentIntensityAnalyzer()

    vader_score = vader.polarity_scores(text)

    return vader_score['compound']


def get_sentiments(list_of_tickers: list, days_back: int = 3):
    sentiments = {}
    past_date = datetime.now() - timedelta(days=days_back)

    list_of_tickers = [i.strip() for i in list_of_tickers]

    for ticker in list_of_tickers:
        if '.' in ticker:
            ticker = ticker.replace('.', '-')

        stock = fvf(ticker)

        if '-' in ticker:
            ticker = ticker.replace('-', '.')

        news = pandas.DataFrame(stock.ticker_news())
        news.set_index('Date', inplace=True)
        news = news[:past_date]

        average_score = 0.0
        for i in range(len(news)):
            article_score = get_article(news['Link'][i], news['Title'][i])
            average_score += article_score

        if len(news) > 0:
            average_score = average_score / len(news)

        sentiments[ticker] = average_score

        print(f"\t{ticker}: {average_score}")

        time.sleep(1)

    return sentiments


def adjust_sentiments(averaged_sentiments: dict):
    adjusted_sentiment = {}

    highest = max(averaged_sentiments.values())
    lowest = min(averaged_sentiments.values())

    for k, v in averaged_sentiments.items():
        if v > 0:
            adjusted_sentiment[k] = v / highest
        elif v < 0:
            adjusted_sentiment[k] = (v - lowest) / lowest
        else:
            adjusted_sentiment[k] = v

    return adjusted_sentiment


def average_sentiments(sentiments: dict):
    average_sentiment = sum([i for i in sentiments.values()]) / len(sentiments)
    averaged_sentiment = {}
    for k, v in sentiments.items():
        averaged_sentiment[k] = 0.00 if abs(v) < 0.00001 else v - (average_sentiment / 2)

    return averaged_sentiment


def begin():
    with open('tickers.txt', 'r') as f:
        tickers = f.readlines()

    sentiments = get_sentiments(tickers, days_back=2)
    averaged_sentiments = average_sentiments(sentiments)
    adjusted_sentiments = adjust_sentiments(averaged_sentiments)

    return sentiments


