import json
import numpy as np
import time
import datetime
from categories import Categories
from stateOneHotEncoding import State
from Regression import Regression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords

def createBagOfwords(train_review_path, test_review_path):
    with open(train_review_path) as f:
        content = [json.loads(line) for line in f]

    train_review_text = ["" for x in range(len(content))]
    train_review_rating = ["" for x in range(len(content))]
    for index in range(len(content)):
        train_review_text[index] = content[index]["text"]
        train_review_rating[index] = content[index]["stars"]

    train_review_rating = np.array(train_review_rating);

    with open(test_review_path) as f:
        content = [json.loads(line) for line in f]

    test_review_text = ["" for x in range(len(content))]
    test_review_rating = ["" for x in range(len(content))]
    for index in range(len(content)):
        test_review_text[index] = content[index]["text"]
        test_review_rating[index] = content[index]["stars"]

    test_review_rating = np.array(test_review_rating)

    stopWords = stopwords.words('english')
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = stopWords, max_features = 5000)

    train_data_features = vectorizer.fit_transform(train_review_text)
    test_data_features = vectorizer.transform(test_review_text)

    #tfidf_transformer = TfidfTransformer()
    #train_data_features = tfidf_transformer.fit_transform(train_data_features)
    #test_data_features = tfidf_transformer.fit(test_data_features)

    train_data_features = train_data_features.toarray()
    test_data_features = test_data_features.toarray()

    return train_data_features, test_data_features, train_review_rating, test_review_rating

def createBusinessDict(business):
    d = {}
    for b in business:
        d[b["business_id"]] = {"stars": b["stars"], "review_count": b["review_count"], "state": b["state"], "categories": b["categories"]}
    return d

def createUserDict(user):
    d = {}
    for u in user:
        d[u["user_id"]] = {"average_stars": u["average_stars"], "review_count": u["review_count"], "fans": u["fans"], "yelping_since": u["yelping_since"], "elite": u["elite"], "compliments": u["compliments"], "friends": u["friends"]}
    return d

def getYelpDay(startDate, reviewDate):
    return (reviewDate-startDate).days

def isElite(currentYear, eliteYear):
    return currentYear in eliteYear

def getTotalCompliments(compliments):
    sum = 0
    for c in compliments:
        sum = sum + compliments[c]
    return sum

def generateFeatures(review_data, business_data, user_data):
    with open(review_data) as f:		# select review dataset here
        review = [json.loads(line) for line in f]

    with open(business_data) as f:
        business = [json.loads(line) for line in f]

    with open(user_data) as f:
        user = [json.loads(line) for line in f]

    businessDict = createBusinessDict(business)
    userDict = createUserDict(user)

    bizDict = {}
    usrDict = {}

    state = State(business)
    categories = Categories()
    x = []
    y = []

    for r in review:
        b = businessDict[r["business_id"]]
        u = userDict[r["user_id"]]

        # business
        if r["business_id"] in bizDict:
            biz = bizDict[r["business_id"]]
        else:
            b_state = state.getOneHotEncodingState(b["state"])
            b_cat = categories.getOneHotEncodingCategoriesFromListOfTitles(b["categories"])
            #biz = [b["stars"], b["review_count"]] + b_state + b_cat
            biz = [b["review_count"]] + b_state + b_cat
            bizDict[r["business_id"]] = biz

        # user
        if r["user_id"] in usrDict:
            usr = usrDict[r["user_id"]]
        else:
            startDate = datetime.datetime.strptime(u["yelping_since"] + "-01", "%Y-%m-%d").date()
            reviewDate = datetime.datetime.strptime(r["date"], "%Y-%m-%d").date()
            u_day = getYelpDay(startDate, reviewDate)
            u_elite = isElite(reviewDate.year, u["elite"])
            u_compliments = getTotalCompliments(u["compliments"])
            #urs = [u["average_stars"], u["review_count"], u["fans"], len(u["friends"]), u_day, u_elite, u_compliments]
            urs = [u["review_count"], u["fans"], len(u["friends"]), u_day, u_elite, u_compliments]
            usrDict[r["user_id"]] = urs

        x.append(biz + urs)
        #x.append(biz)
        #x.append(urs)
        #y.append(r["stars"])

    X = np.array(x)
    #Y = np.array(y)

    return X

###################################################################################################################

np.set_printoptions(threshold=np.inf)
tstart = time.time()
train_review_path = "yelp_academic_dataset_review4.json"
test_review_path = "yelp_academic_dataset_review5.json"
business_path = "yelp_academic_dataset_business.json"
user_path = "yelp_academic_dataset_user.json"

train_bow_features, test_bow_features, train_review_rating, test_review_rating = createBagOfwords(train_review_path, test_review_path)

train_usrbiz_features = generateFeatures(train_review_path, business_path, user_path)
test_usrbiz_features = generateFeatures(test_review_path, business_path, user_path)

# train_features = np.column_stack((train_bow_features,train_usrbiz_features))
# test_features = np.column_stack((test_bow_features,test_usrbiz_features))
##################################################################################################################
reg = Regression()

for model in ['bayesianRegression','ridgeRegression','linearRegression','kernelRidgeRegressionPoly','svmRegressionRbf','nearestNeighbour']:
#for model in ['ridgeRegression']:
    start = time.time()
    reg_bow = getattr(reg, model)(train_bow_features,train_review_rating,'bow')
    predicted_test_bow = reg_bow.predict(test_bow_features)
    predicted_train_bow = reg_bow.predict(train_bow_features)

    reg_usrbiz = getattr(reg, model)(train_usrbiz_features,train_review_rating,'usrbiz')
    predicted_test_usrbiz = reg_usrbiz.predict(test_usrbiz_features)
    predicted_train_usrbiz = reg_usrbiz.predict(train_usrbiz_features)

    #lr = getattr(reg, model)(np.column_stack((predicted_train_bow, predicted_train_usrbiz)),train_review_rating,'combine')
    lr = reg.linearRegression(np.column_stack((predicted_train_bow, predicted_train_usrbiz)),train_review_rating,'combine')
    predicted_final = lr.predict(np.column_stack((predicted_test_bow, predicted_test_usrbiz)))

    print('Model: %s'%model)
    print("Bag Of words")
    reg.performance(predicted_test_bow,test_review_rating)
    print("User Business")
    reg.performance(predicted_test_usrbiz,test_review_rating)
    print("Combined")
    reg.performance(predicted_final,test_review_rating)
    end = time.time()
    print("Time elapsed %s : %s" %(model,end-start))
    print('#############################################################################')
    ###################################################################################################################

# for model in ['kernelRidgeRegressionPoly','bayesianRegression','ridgeRegression','linearRegression','svmRegressionRbf']:
#     start = time.time()
#     #reg_bow = reg.svmRegressionRbf(train_bow_features,train_review_rating,'bow')
#     reg_bow = getattr(reg, model)(train_features,train_review_rating,'bow')
#     predicted_test = reg_bow.predict(test_features)
#
#     print('Model: %s'%model)
#     print("Combined")
#     reg.performance(predicted_test,test_review_rating)
#     end = time.time()
#     print("Time elapsed %s : %s" %(model,end-start))
#     print('#############################################################################')
tend = time.time()
print("Total Time Elapsed:%s" %(tend-tstart))