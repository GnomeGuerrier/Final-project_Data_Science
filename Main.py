import tweepy
from textblob import TextBlob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Twitter API credentials
api_key = '*****'
api_secret = '*****'
access_token = '*****-*****'
access_token_secret = '******'

# Authenticate with Twitter
auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Specific Hashtags and Keywords for the Football Game
team1 = "#Team1"
team2 = "#Team2"
match_hashtag = "#MatchHashtag"

# Game Result (Manually inputted or fetched from another source)
game_result = "Team1 wins"  # Example: "Team1 wins", "Team2 wins", "Draw"

# Time Frame for Before and After the Game
before_game_time = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')  # Example: 24 hours before now
after_game_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Fetching Tweets Before the Game
before_game_tweets = api.search_tweets(q=f"{team1} OR {team2} OR {match_hashtag}", count=100, until=before_game_time)

# Fetching Tweets After the Game
after_game_tweets = api.search_tweets(q=f"{team1} OR {team2} OR {match_hashtag}", count=100, since=after_game_time)

# Function for Sentiment Analysis
def analyze_sentiments(tweets):
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
    return sentiments

# Sentiment Analysis
before_sentiments = analyze_sentiments(before_game_tweets)
after_sentiments = analyze_sentiments(after_game_tweets)

# Plotting
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
axes[0].bar(before_sentiments.keys(), before_sentiments.values())
axes[0].set_title('Sentiments Before Game')
axes[0].set_xlabel('Sentiments')
axes[0].set_ylabel('Number of Tweets')

axes[1].bar(after_sentiments.keys(), after_sentiments.values())
axes[1].set_title('Sentiments After Game')
axes[1].set_xlabel('Sentiments')

plt.suptitle(f'Sentiment Analysis of Tweets for {match_hashtag} ({game_result})')
plt.show()
