from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import string
import Plots
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

train_1 = fetch_20newsgroups(subset='train', categories = categories, shuffle = True, random_state=42)

#print("\n".join(train_1.data[0].split("\n")[:3]))

length = len(train_1.data)
#print("Total records = ",len(train_1.data))


#print(len(train_1))
#print(train_1.target_names)

#tick_list = list(range(len(categories)))
#print(tick_list)
category_count = {}

computer_count = 0
recreational_count = 0
lmtzr = WordNetLemmatizer()

#print(string.punctuation)

for i in range(0,length):
    category = train_1.target_names[train_1.target[i]]
    if(category in category_count):
        counter = category_count[category]
        counter = counter + 1
    else:
        counter = 1
    category_count[category] = counter
    if("comp" in category):
        computer_count = computer_count + 1
    else:
        recreational_count = recreational_count + 1


print("Computer category count= "+str(computer_count))
print("Recreational category count= "+str(recreational_count))
Plots.barPlot(category_count)

####################################################################################
#Removing punctuations
#Tokenize string
stop = stopwords.words('english')
terms = []

def cleanDoc(doc):
    #print("Original\n",doc)
    cleaned =""
    for i in word_tokenize(doc.lower()):
        if i not in stop:
            root = lmtzr.lemmatize(i)
            if root not in string.punctuation:
                cleaned = cleaned+ " "+root
    return cleaned
    #print("CLEANED\n", cleaned)


for i in range(0, length):
    cleaned_doc = cleanDoc(train_1.data[i])
    train_1.data[i] = cleaned_doc



vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(train_1.data)
print("Term Count = ",vectors.shape[1])






#print("\n".join(train_1.data[2].split("\n")[:10]))

#print(text.ENGLISH_STOP_WORDS)


#print(list(category_count.keys()))
#Plots.histogram(list(category_count.values()), "Categories", "Frequency","Plot 1","red", 'File1', list(range(len(categories))),train_1.target_names)
# Plots.linePlot(list(category_count.keys()), list(category_count.values()),"Categories","Frequency","Plot 1","red", 'File1',list(range(len(categories))),train_1.target_names)