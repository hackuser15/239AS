import os
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

def getHashTagStats(hashtag):
    tweet_path = "tweet_data/tweets_#" + hashtag + ".txt"
    abs_tweet_path = os.path.join(script_dir, tweet_path)
    n_followers = 0.0
    n_retweets = 0.0
    n_tweets = 0.0
    d = {}

    start_time = datetime.datetime.now()
    end_time = datetime.datetime.fromtimestamp(datetime.MINYEAR)
    user_ids = set([])
    for line in open(abs_tweet_path, encoding="utf8"):
        tweet=json.loads(line)
        n_tweets += 1
        n_retweets += tweet['metrics']['citations']['data'][0]['citations']
        tweet_time = datetime.datetime.fromtimestamp(tweet['firstpost_date'])
        if tweet_time < start_time:
            start_time = tweet_time
        if tweet_time > end_time:
            end_time = tweet_time
        user_id = tweet['tweet']['user']['id']
        followers = tweet['author']['followers']
        if user_id not in user_ids:
            user_ids.add(user_id)
            n_followers += followers
        if hashtag in ('nfl','superbowl'):
            tweet_time = tweet['firstpost_date']
            tweet_time = datetime.datetime.fromtimestamp(tweet_time)
            hour_key = datetime.datetime(tweet_time.year, tweet_time.month, tweet_time.day, tweet_time.hour, 0, 0)
            if hour_key not in d:
                d[hour_key] = 1
            else:
                d[hour_key] += 1
    print('-----------------------------------------')
    print('Statistics for %s:'%hashtag)
    print('-----------------------------------------')
    print('First tweet time: '+ start_time.strftime('%Y-%m-%d %H:%M:%S'))
    print('Last tweet time: '+ end_time.strftime('%Y-%m-%d %H:%M:%S'))
    n_hours = int((end_time - start_time).total_seconds()/3600 + 0.5)

    # print("total number of hours: ", n_hours)
    # print("total number of users posting this hashtag: ", len(user_ids))
    # print("total number of tweets containing this hashtag: ", n_tweets)
    # print("total number of retweets: ", n_retweets)

    print("Average number of tweets per hour: ", n_tweets/n_hours)
    print("Average number of followers of users posting the tweets: ", n_followers/len(user_ids))
    print("Average number of retweets: ", n_retweets/n_tweets)

    if hashtag in ('nfl','superbowl'):
        plotHistogram(d, hashtag)

def plotHistogram(tweetTimeDict, hashtag):
    start_time = min(tweetTimeDict.keys())
    end_time = max(tweetTimeDict.keys())

    tweets_per_hour = []

    cur = start_time
    while cur <= end_time:
        if cur in tweetTimeDict:
            tweets_per_hour.append(tweetTimeDict[cur])
        else:
            tweets_per_hour.append(0)

        cur += timedelta(hours=1)

    plt.figure(figsize=(20, 8))
    plt.title("Number of Tweets per hour for #" + hashtag)
    plt.ylabel("Number of tweets")
    plt.xlabel("Timeline")
    plt.bar(range(len(tweets_per_hour)), tweets_per_hour, width=1.5, color='b')
    plt.savefig('Histogram_#'+hashtag+'.png')


def genTrainData(hashtag):
    tweet_path = "tweet_data/tweets_#" + hashtag + ".txt"
    abs_tweet_path = os.path.join(script_dir, tweet_path)

    train_data = []
    train_label = []
    firstIteration = True
    # Tweet count, Retweet count, Followers Count, Max Followers, Time
    r = [0, 0, 0, 0, -1]
    for line in open(abs_tweet_path, encoding="utf8"):
        tweet = json.loads(line)
        retweets = tweet['metrics']['citations']['data'][0]['citations']
        followers = tweet['author']['followers']
        tweet_time = tweet['firstpost_date']
        tweet_time = datetime.datetime.fromtimestamp(tweet_time)
        hour_key = datetime.datetime(tweet_time.year, tweet_time.month, tweet_time.day, tweet_time.hour, 0, 0)

        if firstIteration==True:
            cur_hour = hour_key
            firstIteration = False
        while(cur_hour != hour_key):
            train_data.append(r)
            train_label.append(r[0])
            cur_hour += timedelta(hours=1)
            r = [0, 0, 0, 0, cur_hour.hour]

        r[0] += 1                                       #Tweet count
        r[1] += retweets                                #Number of retweets
        r[2] += followers                               #Number of Followers
        r[3] = followers if followers > r[3] else r[3]  #Max Followers
        r[4] = hour_key.hour                            #Hour of Day
    train_data.append(r)
    train_label.append(r[0])
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    train_label = np.roll(train_label, -1)
    train_data = train_data[0:-1,:]
    train_label = train_label[0:-1]
    return train_data, train_label