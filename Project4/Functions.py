import os
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import words

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


def genTrainingData(hashtag, newFeatures = False, asNumpy = True):
    tweet_path = "tweet_data/tweets_#" + hashtag + ".txt"
    abs_tweet_path = os.path.join(script_dir, tweet_path)

    if(newFeatures == True):
        features = ["NumberOfFriends","NumberOfHashtags","NumberOfUsers","NumberOfFav","AvgTweetLength"]
    else:
        features = ["TweetCount","NumberOfRetweets","NumberOfFollowers","MaxFollowers","HourOfDay"]

    train_data = []
    train_label = []
    users = set()
    len_tweet = []
    n_tweets = 0
    firstIteration = True
    # Tweet count, Retweet count, Followers Count, Max Followers, Time
    r = [0, 0, 0, 0, -1]
    for line in open(abs_tweet_path, encoding="utf8"):
        tweet = json.loads(line)
        n_tweets += 1
        retweets = tweet['metrics']['citations']['data'][0]['citations']
        followers = tweet['author']['followers']
        tweet_time = tweet['firstpost_date']
        friends = tweet['tweet']['user']['friends_count']
        n_hashtags = len(tweet['tweet']['entities']['hashtags'])
        users.add(tweet['tweet']['user']['id'])
        n_fav = tweet['tweet']['user']['favourites_count']
        len_tweet.append(len(tweet['tweet']['text']))
        tweet_time = datetime.datetime.fromtimestamp(tweet_time)
        hour_key = datetime.datetime(tweet_time.year, tweet_time.month, tweet_time.day, tweet_time.hour, 0, 0)

        if firstIteration==True:
            cur_hour = hour_key
            firstIteration = False
        while(cur_hour != hour_key):
            train_data.append(r)
            train_label.append(n_tweets)
            n_tweets = 0
            cur_hour += timedelta(hours=1)
            if newFeatures == False:
                r = [0, 0, 0, 0, cur_hour.hour]
            else:
                users.clear()
                len_tweet = []
                r = [0, 0, 0, 0, 0]
        if newFeatures == False:
            r[0] = n_tweets                                   #Tweet count
            r[1] += retweets                                  #Number of retweets
            r[2] += followers                                 #Number of Followers
            r[3] = followers if followers > r[3] else r[3]    #Max Followers
            r[4] = hour_key.hour                              #Hour of Day
        else:
            r[0] += friends                                   #No of friends
            r[1] += n_hashtags                                #Number of hashtags
            r[2] = len(users)                                 #No of users posting
            r[3] += n_fav                                     #No of favourites
            r[4] = 0 if not len_tweet else np.mean(len_tweet) #Average length of tweets
    train_data.append(r)
    train_label.append(n_tweets)
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    train_label = np.roll(train_label, -1)
    train_data = train_data[0:-1,:]
    train_label = train_label[0:-1]
    if(asNumpy == False):
        train_data = pd.DataFrame(data=train_data, index=range(len(train_data)), columns=features)
        train_label = pd.DataFrame(data=train_label, index=range(len(train_label)), columns=["NumberOfTweets"])
    if(newFeatures == False and asNumpy == False):
        train_data,_,_ = one_hot_dataframe(train_data,['HourOfDay'], replace=True)
    return train_data, train_label

def one_hot_dataframe(data, cols, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
        Returns a 3-tuple comprising the data, the vectorized data,
        and the fitted vectorizor.
    """
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col].astype(str)) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData.astype(float))
    return (data, vecData, vec)

def getReadabilityScore(tweet):
    w1 = tweet.split(" ")
    ASL1 = len(w1)
    AOV1 = 0
    l = 0
    for w in w1:
        l+=len(w)
        if(w not in words.words()):
            AOV1+=1
    ASW1 = l/float(ASL1)
    S1 = 206.835 - (1.015*ASL1) - (84.6*ASW1)- (10.5*AOV1)
    return S1