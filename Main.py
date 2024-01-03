import tweepy
import requests
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
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

def SentimentDistribution(sentiments_over_time):
    #Allows to see the Sentiment distribution on a graph
    overall_sentiments = pd.Series(sentiments_over_time).value_counts().sort_index()
    plt.pie(overall_sentiments, labels=['Negative', 'Neutral', 'Positive'], colors=['red', 'blue', 'green'], autopct='%1.1f%%')
    plt.title('Overall Sentiment Distribution')
    plt.show()

def SentimentOverTime(df_time_based):
    #Allows to plot the average sentiment over time
    df_time_grouped = df_time_based.groupby(df_time_based['Time'].dt.hour).mean()
    plt.plot(df_time_grouped.index, df_time_grouped['Sentiment'], marker='o', linestyle='-')
    plt.title('Sentiment Over Time')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Sentiment Score')
    plt.ylim(0, 2)
    plt.show()
    
def PrePostSentiment (before_counts_new,after_counts_new):
    #Allows to plot thepre and post sentiment of a game
    categories = ['Pre-Game', 'Post-Game']
    plt.bar(categories, [before_counts_new.sum(), after_counts_new.sum()], color='grey')
    plt.bar(categories, [before_counts_new[2], after_counts_new[2]], color='green', label='Positive')
    plt.bar(categories, [before_counts_new[1], after_counts_new[1]], color='blue', bottom=[before_counts_new[2], after_counts_new[2]], label='Neutral')
    plt.bar(categories, [before_counts_new[0], after_counts_new[0]], color='red', bottom=[before_counts_new[1]+before_counts_new[2], after_counts_new[1]+after_counts_new[2]], label='Negative')
    plt.title('Comparison of Sentiment Pre- and Post-Game')
    plt.legend()
    plt.show()



def UserEngagement(df_time_based):
    tweet_counts = df_time_based['Time'].dt.hour.value_counts().sort_index()
    plt.bar(tweet_counts.index, tweet_counts.values, color='orange')
    plt.title('User Engagement Levels Over Time')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Tweets')
    plt.show()

def RandomForestModel(sentiment_texts,game_outcomes):
    data = pd.DataFrame({
    'text': sentiment_texts,
    'game_outcome': game_outcomes
    })

    # Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['text'])

    # Labels
    y = data['game_outcome']

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Choosing a model (Random Forest in this case) and reducing its complexity
    classifier = RandomForestClassifier(random_state=0, max_depth=3, n_estimators=10)

    # Training the model
    classifier.fit(X_train, y_train)

    # Predictions
    y_pred = classifier.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print('Accuracy:', accuracy)
    print('Classification Report:\n', report)

    # Visualization: comparison of actual and predicted outcomes
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Outcome', alpha=0.5)
    plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Outcome', alpha=0.5)
    plt.title('Actual vs Predicted Game Outcomes')
    plt.ylabel('Game Outcome')
    plt.legend()
    plt.show()

        
    # Predict probabilities for the positive outcome (which should be the second column if 'win' is encoded as 1)
    y_proba = classifier.predict_proba(X_test)

    # Calculate ROC curve and ROC AUC for the positive class ('win')
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1], pos_label='win')
    roc_auc = auc(fpr, tpr)

    # Calculate precision-recall curve and average precision for the positive class ('win')
    precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1], pos_label='win')
    average_precision = average_precision_score(y_test, y_proba[:, 1], pos_label='win')

    # Plot ROC curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Plot Precision-Recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='green', lw=2, label='Precision-Recall curve (area = %0.2f)' % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.show()




def main():
    # User input to select data source
    data_source = input("Select data source (1 for CSV file, 2 for online): ")

    if data_source == '1':
        # Read from CSV file
        file_path = input("Enter the path to the CSV file: ")
        tweets_df = read_csv_file(file_path)
        before_game_tweets = tweets_df[tweets_df['Date'] < 'game_day_time']  # Replace 'game_day_time' with actual game day this is important
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