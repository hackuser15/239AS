#https://ucla.box.com/s/nv9td9kvvfvg3tya0dlvbs1kn5o87gmv
#Download tweet data from above mentioned url and folder in tweet_data in project directory

import time
from Project4.Functions import *
import statsmodels.api as sm
import matplotlib.pyplot as plt

start_time = time.time()

hashtags = ["gohawks","gopatriots","nfl","patriots","sb49","superbowl"]
# hashtags = ["gohawks"]

#Q.1
print("--------------Q1----------------")
for hashtag in hashtags:
    getHashTagStats(hashtag)

#Q2
print("------------Q2------------------")
for hashtag in hashtags:
    train_data, train_label = genTrainingData(hashtag)
    model = sm.OLS(train_label, train_data.astype(float))
    results = model.fit()
    print('-----------------------------------------')
    print('Linear Regression Statistics for %s:'%hashtag)
    print('-----------------------------------------')
    print(results.summary())

    with open("./Results/Stats/Linear_Regression_Result_#"+hashtag+".txt", 'w') as fp:
        fp.write(str(results.summary()))
        fp.close()

#Q3
print("------------Q3------------------")
for hashtag in hashtags:
    train_data, train_label = genTrainingData(hashtag, newFeatures = True)
    # np.savetxt('test2.txt', train_data, fmt = '%-7.2f')
    model = sm.OLS(train_label, train_data)
    results = model.fit()
    print('-----------------------------------------')
    print('Linear Regression Statistics for %s:'%hashtag)
    print('-----------------------------------------')
    print(results.summary())

    with open("./Results/Stats/Linear_Regression_Result_New_#"+hashtag+".txt", 'w') as fp:
        fp.write(str(results.summary()))
        fp.close()

    for feature in ['NumberOfFriends','NumberOfHashtags','NumberOfFav']:
        plt.scatter(train_data[feature],train_label, marker='o')
        plt.xlabel(feature)
        plt.ylabel('Number of tweets')
        plt.title('Number of tweets vs '+ feature)
        plt.savefig('./Results/Plots/'+hashtag+'-'+feature+'.png')
        plt.clf()

print("--- %s seconds ---" % (time.time() - start_time))
