import pandas as pd
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from textblob import TextBlob
from fredapi import Fred
import time

# ----------------------------
# Step 1: Set Your API Keys
# ----------------------------
NEWS_API_KEY = '20c96e4f99d84b32959a5dfe25370049'  # Replace with your key
FRED_API_KEY = 'ce869862952bb808a43b95a7d84f915c'

# ----------------------------
# Step 2: Initialize Clients
# ----------------------------
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
fred = Fred(api_key=FRED_API_KEY)

# ----------------------------
# Step 3: Get Weekly Macroeconomic Data (2020–2025)
# ----------------------------
interest = fred.get_series('DGS10')  # 10-Year Treasury
cpi = fred.get_series('CPIAUCSL')    # CPI Index

df_interest = interest.to_frame(name='InterestRate').resample('W').mean()
df_cpi = cpi.to_frame(name='CPI').resample('W').mean()

# ----------------------------
# Step 4: Get Weekly News Sentiment (from April 5, 2025)
# ----------------------------
start_date = datetime(2025, 4, 5)
end_date = datetime.today()
current_date = start_date

news_data = []

while current_date < end_date:
    next_date = current_date + timedelta(days=6)
    from_str = current_date.strftime('%Y-%m-%d')
    to_str = next_date.strftime('%Y-%m-%d')

    try:
        print(f"⏳ Fetching: {from_str} → {to_str}")
        articles = newsapi.get_everything(
            q='semiconductor',
            from_param=from_str,
            to=to_str,
            language='en',
            sort_by='relevancy',
            page_size=100
        )['articles']

        sentiments = []
        for article in articles:
            if article.get('title'):
                polarity = TextBlob(article['title']).sentiment.polarity
                sentiments.append(polarity)

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        news_data.append({'week': from_str, 'sentiment': avg_sentiment})
        print(f"✅ {from_str}: {avg_sentiment:.3f} ({len(sentiments)} titles)")

    except Exception as e:
        print(f"⚠️ NewsAPI Error {from_str}: {e}")
        if "rateLimited" in str(e):
            print("⏸️ Sleeping for 12 hours to reset NewsAPI limit...")
            time.sleep(12 * 60 * 60)  # sleep 12 hours
            continue
        news_data.append({'week': from_str, 'sentiment': None})

    current_date = next_date + timedelta(days=1)
    time.sleep(1)  # avoid hammering API

# ----------------------------
# Step 5: Merge and Export
# ----------------------------
df_sentiment = pd.DataFrame(news_data)
df_sentiment['week'] = pd.to_datetime(df_sentiment['week'])
df_sentiment.set_index('week', inplace=True)

external_factors = df_interest.join(df_cpi).join(df_sentiment)
external_factors.to_csv('external_factors_weekly_.csv')

print("✅ Exported: external_factors_weekly_.csv")
