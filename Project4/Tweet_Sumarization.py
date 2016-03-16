from sumy.parsers.plaintext import PlaintextParser #We're choosing a plaintext parser here, other parsers available for HTML etc.
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer #We're choosing Lexrank, other algorithms are also built in
import os
import json
import re
#import enchant
import datetime
from Project4.Functions import *
from collections import OrderedDict


hashtags = ["gohawks","gopatriots","nfl","patriots","sb49","superbowl"]

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

for hashtag in hashtags:
    tweet_path = "tweet_data/tweets_#" + hashtag + ".txt"
    abs_tweet_path = os.path.join(script_dir, tweet_path)
    f1 = open("Week1.txt","w")
    f2 = open("Week2.txt","w")
    f3 = open("Week3.txt","w")

    c1 = 0
    c2 = 0
    c3 = 0
    max_tweets = 1000
    for line in open(abs_tweet_path, encoding="utf8"):
            tweet=json.loads(line)
            text = tweet['tweet']['text']
            lang = tweet['tweet']['lang']

            #Remove non-english tweets
            if(lang !='en'):
                continue
            #Removing hashtag and url's and any referenced users
            text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())

            #f.write(text+".\n")
            tweet_time = tweet['firstpost_date']
            tweet_time = datetime.datetime.fromtimestamp(tweet_time)

            if(tweet_time <datetime.datetime(2015,2,1,8,0,0) and c1 <=max_tweets):
                f1.write(text+".\n")
                c1+=1
            elif(tweet_time >= datetime.datetime(2015,2,1,8,0,0) and tweet_time <= datetime.datetime(2015,2,1,20,0,0)and c2 <=max_tweets):
                f2.write(text+".\n")
                c2+=1
            elif(tweet_time > datetime.datetime(2015,2,1,20,0,0) and c3<=max_tweets):
                f3.write(text+".\n")
                c3+=1
            if(c1>max_tweets and c2>max_tweets and c3>max_tweets):
                break
     #3 files created for current hashtag
     #Finding 3 week summary for current hashtag
    print("Summary of Tweets for Hashtag = "+hashtag)
    for file in range(1,4):
        if(file == 1):
            message = "BEFORE GAME DAY"
        elif(file == 2):
            message = "ON GAME DAY"
        else:
            message = "AFTER GAME DAY"
        print(message)
        print("-"*100)
        file = "Week"+str(file)+".txt" #name of the plain-text file
        parser = PlaintextParser.from_file(file, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, 20) #Summarize the document with 20 tweets
        tweet_list = []
        top_tweet_count = 0
        ranked_tweets = {}
        for sentence in summary:
            #Filter out tweets with exact same message
            orig = str(sentence)
            t = orig.lower()
            t = ''.join(sorted(t))
            l = [x for x in tweet_list if t == x]
            if(len(l) > 0):
                continue
            score = getReadabilityScore(orig)
            ranked_tweets[orig] = score
            #print(orig,":",score)
            #top_tweet_count+=1
            tweet_list.append(t)
        count = 0
        for w in sorted(ranked_tweets, key=ranked_tweets.get, reverse=True):
            if(count <= 10):
                print(w, ranked_tweets[w])
                count+=1
            else:
                break
        print("-"*50)

    f1.close()
    f2.close()
    f3.close()



