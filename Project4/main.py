#https://ucla.box.com/s/nv9td9kvvfvg3tya0dlvbs1kn5o87gmv
#Download tweet data from above mentioned url and folder in tweet_data in project directory

import time
from Project4.Functions import *
import statsmodels.api as sm

start_time = time.time()

hashtags = ["gohawks","gopatriots","nfl","patriots","sb49","superbowl"]

#Q.1
print("--------------Q1----------------")
for hashtag in hashtags:
    getHashTagStats(hashtag)

#Q2
print("------------Q2------------------")
train_data, train_label = genTrainData("gohawks")
for hashtag in hashtags:
    train_data, train_label = genTrainData(hashtag)
    model = sm.OLS(train_label, train_data)
    results = model.fit()
    print('-----------------------------------------')
    print('Linear Regression Statistics for %s:'%hashtag)
    print('-----------------------------------------')
    print(results.summary())

    with open("Linear_Regression_Result_#"+hashtag+".txt", 'w') as fp:
        fp.write(str(results.summary()))
        fp.close()

print("--- %s seconds ---" % (time.time() - start_time))
