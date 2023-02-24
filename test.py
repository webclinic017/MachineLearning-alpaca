import time
from finvizfinance.quote import finvizfinance as fvf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup


def p_to_text(x):
    return x.get_text()


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
            texts.append(p_to_text(i))

        line = (' '.join(texts)).strip()
    except:
        print("EXCEPTION")
        return analyze_article(title)

    print(analyze_article(line))
    return analyze_article(line)


def analyze_article(text):
    vader = SentimentIntensityAnalyzer()

    vader_score = vader.polarity_scores(text)

    return vader_score['compound']


if __name__ == '__main__':
    start = time.time()

    with open('tickers.txt', 'r') as f:
        list_of_tickers = f.readlines()

    sentiments = {}
    days_back = 2
    current_date = datetime.now()
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

    average_sentiment = sum([i for i in sentiments.values()]) / len(sentiments)
    adjusted_sentiment = {}

    print(f"average sentiment: {average_sentiment}")
    for k, v in sentiments.items():
        adjusted_sentiment[k] = 0.00 if abs(v) < 0.00001 else v - (average_sentiment / 2)

    highest = max(adjusted_sentiment.values())
    lowest = min(adjusted_sentiment.values())

    for k, v in adjusted_sentiment.items():
        if v > 0:
            adjusted_sentiment[k] = v / highest
        elif v < 0:
            adjusted_sentiment[k] = (v - lowest) / lowest

    print(sentiments)
    print()
    print(adjusted_sentiment)
    end = time.time()

    print(f"time elapsed: {end-start}")
