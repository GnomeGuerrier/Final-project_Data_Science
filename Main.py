import tweepy
import requests
import pandas as pd
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

file_path = 'path_to_your_csv_file.csv'

# Function for Sentiment Analysis
def analyze_sentiments(tweets, from_csv=False):
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    for tweet in tweets:
        if from_csv:
            # Use the polarity provided in the CSV file
            polarity = tweet['Polarity']
        else:
            # Analyze tweet text using TextBlob
            analysis = TextBlob(tweet.text)
            polarity = analysis.sentiment.polarity

        if polarity > 0:
            sentiments["positive"] += 1
        elif polarity == 0:
            sentiments["neutral"] += 1
        else:
            sentiments["negative"] += 1
    return sentiments

def getGameResult(team1,team2):
    # API credentials and endpoints 
    api_endpoint = "https://api.football-data.org/v2/matches?status=FINISHED"
    api_key = "*******"

    # Headers for API request
    headers = {"X-Auth-Token": api_key}

    # Fetching game results
    response = requests.get(api_endpoint, headers=headers)
    data = response.json()


    latest_match = data['matches'][0]
    team1 = latest_match['homeTeam']['name']
    team2 = latest_match['awayTeam']['name']
    team1_goals = latest_match['score']['fullTime']['home']
    team2_goals = latest_match['score']['fullTime']['away']

    if team1_goals > team2_goals:
        game_result = f"{team1} wins"
    elif team2_goals > team1_goals:
        game_result = f"{team2} wins"
    else:
        game_result = "Draw"

    return game_result



def read_csv_file(file_path):
    # Reading the CSV file
    df = pd.read_csv(file_path)
    
    # Renaming columns for clarity
    df.columns = ['Polarity', 'ID', 'Date', 'Query', 'User', 'Text']
    
    return df





def main():
    # User input to select data source
    data_source = input("Select data source (1 for CSV file, 2 for online): ")

    if data_source == '1':
        # Read from CSV file
        file_path = input("Enter the path to the CSV file: ")
        tweets_df = read_csv_file(file_path)
        before_game_tweets = tweets_df[tweets_df['Date'] < 'game_day_time']  # Replace 'game_day_time' with actual game day
        after_game_tweets = tweets_df[tweets_df['Date'] >= 'game_day_time']

        # Analyze sentiments from CSV tweets
        before_sentiments = analyze_sentiments(before_game_tweets.to_dict('records'), from_csv=True)
        after_sentiments = analyze_sentiments(after_game_tweets.to_dict('records'), from_csv=True)
        
        
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

    elif data_source == '2':
        

        # Time Frame for Before and After the Game
        before_game_time = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')  
        after_game_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Fetching Tweets Before the Game
        before_game_tweets = api.search_tweets(q=f"{team1} OR {team2} OR {match_hashtag}", count=100, until=before_game_time)

        # Fetching Tweets After the Game
        after_game_tweets = api.search_tweets(q=f"{team1} OR {team2} OR {match_hashtag}", count=100, since=after_game_time)
        
        game_result = getGameResult("team1","team2")
                
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


        
        
    else:
        print("Invalid input. Please select 1 or 2.")

if __name__ == "__main__":
    main()