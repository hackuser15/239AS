#https://ucla.box.com/s/nv9td9kvvfvg3tya0dlvbs1kn5o87gmv
#Download tweet data from above mentioned url and folder in tweet_data in project directory

import time

from sklearn.ensemble import RandomForestRegressor
from Project4.Functions import *
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

start_time = time.time()

hashtags = ["gohawks","gopatriots","nfl","patriots","sb49","superbowl"]

#Q.1
print("--------------Q1----------------")
for hashtag in hashtags:
    getHashTagStats(hashtag)

#Q2
print("------------Q2------------------")
for hashtag in hashtags:
    train_data, train_label, _, _ = genTrainingData(hashtag,'train', asNumpy=False)
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
print("------------Q3/Q4/Q5------------------")
new_features = ['Number of friends','Number of Hashtags','Number of Users','Number of favourites','Average tweet length']
for hashtag in hashtags:
    print("HASHTAG %s" %hashtag)
    train_data, train_label, second, third = genTrainingData(hashtag, 'train' ,newFeatures = True)
    model = RandomForestRegressor(n_estimators=10)
    model.fit(train_data, train_label)
    score = model.score(train_data,train_label)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("r2 score %.2f:" %score)
    for f in range(train_data.shape[1]):
        print("%d. feature %d %s (%f)" % (f + 1, indices[f],new_features[indices[f]],importances[indices[f]]))
    for col in [indices[0],indices[1],indices[2]]:
        plt.scatter(train_data[:,col],train_label, marker='o')
        plt.xlabel(new_features[col])
        plt.ylabel('Number of tweets')
        plt.title('Number of tweets vs '+ new_features[col])
        plt.savefig(hashtag+'-'+new_features[col]+'.png')
        plt.clf()

    ## Q4
    train_data_before8am = train_data[:second,:]
    train_label_before8am = train_label[:second]
    train_data_between8am8pm = train_data[second:third,:]
    train_label_between8am8pm = train_label[second:third]
    train_data_after8pm = train_data[third:,:]
    train_label_after8pm = train_label[third:]
    test_periods1 = ['sample1_period1','sample4_period1','sample5_period1','sample8_period1']
    test_periods2 = ['sample2_period2','sample6_period2','sample9_period2']
    test_periods3 = ['sample3_period3','sample7_period3','sample10_period3']
    obj = RandomForestRegressor(n_estimators=10)
    KfoldLR(obj,train_data,train_label,"ENTIRE SET")

    obj = RandomForestRegressor(n_estimators=10)
    obj = KfoldLR(obj,train_data_before8am,train_label_before8am,"BEFORE Feb 1, 8 AM")
    # Here we run the sample files of period1 against all hashtags, but it is to be noted we report the predicted value
    # only for the dominant hashtag which is found in the code after this loop.
    for test_file in test_periods1:
        print(test_file)
        train_data, train_label, _, _ = genTrainingData(test_file, 'test',newFeatures = True)
        test_data = train_data[train_data.shape[0]-1]
        test_label = train_label[train_label.shape[0]-1]
        predicted=obj.predict(test_data)
        print("The predicted number of tweets for the next hour for this sample window: %.2f" %predicted)

    obj = RandomForestRegressor(n_estimators=10)
    obj = KfoldLR(obj,train_data_between8am8pm,train_label_between8am8pm,"BETWEEN Feb 1, 8 AM AND 8 PM")
    # Here we run the sample files of period2 against all hashtags, but it is to be noted we report the predicted value
    # only for the dominant hashtag which is found in the code after this loop.
    for test_file in test_periods2:
        print(test_file)
        train_data, train_label, _, _ = genTrainingData(test_file, 'test',newFeatures = True)
        test_data = train_data[train_data.shape[0]-1]
        test_label = train_label[train_label.shape[0]-1]
        predicted=obj.predict(test_data)
        print("The predicted number of tweets for the next hour for this sample window: %.2f" %predicted)

    obj = RandomForestRegressor(n_estimators=10)
    obj = KfoldLR(obj,train_data_after8pm,train_label_after8pm,"AFTER Feb 1, 8 PM")
    # Here we run the sample files of period3 against all hashtags, but it is to be noted we report the predicted value
    # only for the dominant hashtag which is found in the code after this loop.
    for test_file in test_periods3:
        print(test_file)
        train_data, train_label, _, _ = genTrainingData(test_file, 'test',newFeatures = True)
        test_data = train_data[train_data.shape[0]-1]
        test_label = train_label[train_label.shape[0]-1]
        predicted=obj.predict(test_data)
        print("The predicted number of tweets for the next hour for this sample window: %.2f" %predicted)

## Q5
script_dir = os.path.dirname(__file__)
test_periods = ['sample1_period1','sample4_period1','sample5_period1','sample8_period1',
                'sample2_period2','sample6_period2','sample9_period2',
                'sample3_period3','sample7_period3','sample10_period3']
for file in test_periods:
    print(file)
    tweet_path = "test_data/" + file + ".txt"
    abs_tweet_path = os.path.join(script_dir, tweet_path)
    hashtagCount = np.zeros((len(hashtags)),dtype="float32")
    for line in open(abs_tweet_path, encoding="utf8"):
        tweet = json.loads(line)
        text = tweet['highlight']
        for i in range(0,len(hashtags)):
            if hashtags[i] in text.lower():
                hashtagCount[i] += 1
    argm = hashtagCount.argmax()
    print("Dominant hashtag %s" %hashtags[argm])
# The values are predicted in the above loop but we choose the predicted values only for the dominant hashtag.
print("--- %s seconds ---" % (time.time() - start_time))
