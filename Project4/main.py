#https://ucla.box.com/s/nv9td9kvvfvg3tya0dlvbs1kn5o87gmv
#Download tweet data from above mentioned url and folder in tweet_data in project directory

import os
import json
import time
from Project4.Functions import *

start_time = time.time()

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

hashtags = ["gohawks","gopatriots","nfl","patriots","sb49","superbowl"]
hashtag_data = {}

# for hashtag in hashtags:
#     tweet_data = []
#     tweet_path = "tweet_data/tweets_#" + hashtag + ".txt"
#     abs_tweet_path = os.path.join(script_dir, tweet_path)
#     for line in open(abs_tweet_path, encoding="utf8"):
#         tweet_data.append(json.loads(line))
#     hashtag_data[hashtag] = tweet_data

#Q.1
print("--------------Q1----------------")
for hashtag in hashtags:
    tweet_data = []
    tweet_path = "tweet_data/tweets_#" + hashtag + ".txt"
    abs_tweet_path = os.path.join(script_dir, tweet_path)
    for line in open(abs_tweet_path, encoding="utf8"):
        tweet_data.append(json.loads(line))
    hashtag_data[hashtag] = tweet_data
    getHashTagStats(hashtag_data[hashtag], hashtag)

for hashtag in ["nfl","superbowl"]:
    tweet_data = []
    tweet_path = "tweet_data/tweets_#" + hashtag + ".txt"
    abs_tweet_path = os.path.join(script_dir, tweet_path)
    for line in open(abs_tweet_path, encoding="utf8"):
        tweet_data.append(json.loads(line))
    hashtag_data[hashtag] = tweet_data
    plotHistogram(hashtag_data[hashtag], hashtag)

print("--- %s seconds ---" % (time.time() - start_time))
