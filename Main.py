import tweepy
from textblob import TextBlob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Twitter API credentials
api_key = 'pmE3eJn3wAncZboVybljobSPf'
api_secret = 'VsMgrw5IJ3w3dDtQOpnfedvKpVkxGMIvFY3gp3zcWxfQdqjLgR'
access_token = '1621506331906830349-KwapThh9sa9X1sLTsZYl46WtpCRoAx'
access_token_secret = 'V8clCdfEOU2Uo2o1omFzuJ27fcpQjJuoN1LJJDgYgGUkJ'

# Authenticate with Twitter
auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Hashtag and Time Frame
hashtag = "#YourHashtag"
since_time = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')

# Fetching Tweets
tweets = api.search_tweets(q=hashtag, count=2, since=since_time)

# Sentiment Analysis
sentiments = {"positive": 0, "neutral": 0, "negative": 0}
for tweet in tweets:
    analysis = TextBlob(tweet.text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        sentiments["positive"] += 1
    elif polarity == 0:
        sentiments["neutral"] += 1
    else:
        sentiments["negative"] += 1

# Plotting
labels = sentiments.keys()
sizes = sentiments.values()

plt.bar(labels, sizes)
plt.xlabel('Sentiments')
plt.ylabel('Number of Tweets')
plt.title('Sentiment Analysis of Tweets')
plt.show()
