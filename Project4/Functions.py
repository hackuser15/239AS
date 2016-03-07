import datetime
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta

def getHashTagStats(hashtag_data, hashtag):
    n_followers = 0.0
    n_retweets = 0.0
    n_tweets = 0.0

    start_date = datetime.datetime.fromtimestamp(hashtag_data[0]['firstpost_date'])
    end_date = datetime.datetime.fromtimestamp(datetime.MINYEAR)

    user_ids = set([])
    for tweet in hashtag_data:
        n_tweets += 1
        n_retweets += tweet['metrics']['citations']['data'][0]['citations']
        tweet_time = datetime.datetime.fromtimestamp(tweet['firstpost_date'])
        if tweet_time < start_date:
            start_date = tweet_time
        if tweet_time > end_date:
            end_date = tweet_time
        user_id = tweet['tweet']['user']['id']
        followers = tweet['author']['followers']
        if user_id not in user_ids:
            user_ids.add(user_id)
            n_followers += followers
    print('Statistics for %s:'%hashtag)
    print('First tweet time: '+ start_date.strftime('%Y-%m-%d %H:%M:%S'))
    print('Last tweet time: '+ end_date.strftime('%Y-%m-%d %H:%M:%S'))
    n_hours = int((end_date - start_date).total_seconds()/3600 + 0.5)

    # print("total number of hour: ", n_hours)
    # print("total number of users posting this hashtag: ", len(user_ids))
    # print("total number of tweets containing this hashtag: ", n_tweets)
    # print("total number of retweets: ", n_retweets)
    print('-----------------------------------------')
    print("Average number of tweets per hour: ", n_tweets/n_hours)
    print("Average number of followers of users posting the tweets: ", n_followers/len(user_ids))
    print("Average number of retweets: ", n_retweets/n_tweets)
    print('-----------------------------------------')

def plotHistogram(hashtag_data, hashtag):
    d = {}
    for tweet in hashtag_data:
        tweet_time = tweet["firstpost_date"]
        tweet_time = datetime.datetime.fromtimestamp(tweet_time)
        time_key = datetime.datetime(tweet_time.year, tweet_time.month, tweet_time.day, tweet_time.hour, 0, 0)
        if time_key not in d:
            d[time_key] = 1
        else:
            d[time_key] += 1

    start_time = min(d.keys())
    end_time = max(d.keys())

    tweets_per_hour = []

    cur = start_time
    while cur <= end_time:
        if cur in d:
            tweets_per_hour.append(d[cur])
        else:
            tweets_per_hour.append(0)

        cur += timedelta(hours=1)

    plt.figure(figsize=(20, 8))
    plt.title("Number of Tweets per hour for #" + hashtag)
    plt.ylabel("Number of tweets")
    plt.xlabel("Timeline")
    plt.bar(range(len(tweets_per_hour)), tweets_per_hour, width=1.5, color='b')
    plt.savefig('Histogram_#'+hashtag+'.png')