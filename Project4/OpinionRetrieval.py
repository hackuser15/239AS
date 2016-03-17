import os
import json
import re
#import enchant
import datetime
from Project4.Functions import *
from collections import OrderedDict
import csv
import numpy as np
import statsmodels.api as sm
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from nltk.stem.porter import *
from nltk.corpus import stopwords
from collections import Counter
import math

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
hashtags = ["gopatriots"]

output = open("Results/tweet_opinion_score.txt", 'w')

train_data = []
train_label = []
train_tweet = []
train_orig_tweet = []
#Using hashtag gohawks to train the binary classifier

word_dict = {}
stop = stopwords.words('english')
stemmer = PorterStemmer()

for hashtag in hashtags:
    #User mentions(0/1),urls(1/0),follower_count,status_count,friend_count,favorited(liked)(0/1)
    tweet_path = "tweet_data/tweets_#" + hashtag + ".txt"
    abs_tweet_path = os.path.join(script_dir, tweet_path)
    subj_count = 0
    obj_count = 0
    for line in open(abs_tweet_path, encoding="utf8"):
        isSubjective = False
        features = [0,0,0,0,0,0]
        tweet=json.loads(line)
        text_orig = tweet['tweet']['text']
        lang = tweet['tweet']['lang']

        #Removing hashtag and url's and any referenced users
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text_orig).split())


        #Filtering out non-english tweets
        if(lang !='en'):
            continue

        #Check if tweet has any user mentions
        users = tweet['tweet']['entities']['user_mentions']
        if len(users) > 0:
            features[0] = 1
        else:
            features[0] = 0

        #Check if tweet has any URL's
        urls = tweet['tweet']['entities']['urls']
        if(len(urls) > 0):
            features[1] = 1
        else:
            features[1] = 0

        #Follower count
        follower_count = tweet['tweet']['user']['followers_count']
        features[2] = follower_count

        #Number of previous statuses
        statuses_count = tweet['tweet']['user']['statuses_count']
        features[3] = statuses_count

        #Friend count
        friends_count = tweet['tweet']['user']['friends_count']
        features[4] = friends_count

        #Favorited
        favorited_count = tweet['tweet']['favorite_count']
        if(favorited_count > 0):
            features[5] = 1
        else:
            features[5] = 0

        label = 0
        add_to_training = False
        #Assign training label based on Pseudo Subjective Tweet(PST) and Pseudo Objective Tweet(POT)
        #A tweet is assumed to be subjective/opinionated if a tweet has a user mention with a text of atleast 10 characters before it
        if(features[0] == 1):
            for user_mention in users:
                index_list = user_mention["indices"]
                if(index_list[1] > 10):
                    label = 1
                    add_to_training = True
                    isSubjective = True
                    subj_count+=1
                    break
        elif features[1] == 1 and follower_count > 1000 and statuses_count>4000:
            label = 0
            obj_count+=1
            add_to_training = True

        if(add_to_training):
            train_data.append(features)
            train_label.append(label)
            train_tweet.append(text)
            train_orig_tweet.append(text_orig)

        else:
            continue


        for word in text.split(" "):
            if word not in stop:
                root = stemmer.stem(word.lower())
                if root in word_dict:
                    sub_obj_list = word_dict[root]
                else:
                    sub_obj_list = [0,0]
                if isSubjective:
                    sub_obj_list[0]+=1
                else:
                    sub_obj_list[1]+=1
                word_dict[root] = sub_obj_list
            else:
                continue

#print(word_dict)
print("Subjective tweets= ",str(subj_count)," Objective tweets= ",str(obj_count))
#Loop through tweets to find opinionatedness score of tweets
for i in range(0, len(train_tweet)):
    tweet = train_tweet[i]
    tweet_opinion = 0
    tweet_length = len(tweet.split(" "))
    #l is list of unique words
    l = Counter(tweet.split()).most_common()
    for word_tuple in l:
        word = word_tuple[0]
        if word not in stop:
            rel_frequency = word_tuple[1]/float(tweet_length)
            root = stemmer.stem(word.lower())
            sub_obj_list = word_dict[root]
            O11 = sub_obj_list[0]
            O12 = subj_count - O11
            O21 = sub_obj_list[1]
            O22 = obj_count - O21
            O_1 = O11 + O21
            O_2 = O12 + O22
            O1_ = O11 + O12
            O2_ = O21 + O22
            O = O1_ + O2_
            pearson = (math.pow(O11*O22 - O12*O21,2)*O)/float(O1_*O2_*O_1*O_2)
            if(pearson < 5.02):
                continue
            opinion = np.sign((O11/float(O1_)) - (O21/float(O2_)))*pearson
            tweet_opinion+=opinion
    output.write(train_orig_tweet[i]+"::"+str(tweet_opinion)+" "+str(train_label[i])+"\n")
    output.write("\n")


print("Support Vector Machine Analysis")
svc = SVC(kernel='linear', probability=True, random_state=40)
X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.20, random_state=42)

svc.fit(X_train,y_train)
predicted = svc.predict(X_test)
preds = svc.predict_proba(X_test)
print("Accuracy for this classifier:%s\n" % np.mean(predicted == y_test))





