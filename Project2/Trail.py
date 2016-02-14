from nltk.chunk.named_entity import shape
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
import string
import Plots
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import math
import collections


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
#
computer_count = 0
recreational_count = 0
lmtzr = WordNetLemmatizer()

#print(string.punctuation)
print(train_1.target_names)
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


for i in range(0, 10):
    cleaned_doc = cleanDoc(train_1.data[i])
    train_1.data[i] = cleaned_doc



vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(train_1.data)
print("Term Count = ",vectors.shape[1])


##########################################################
train_all = fetch_20newsgroups(subset='train', shuffle = True, random_state=42)
length_total = len(train_all.data)
categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
indices = []
for i in categories:
    indices.append(train_all.target_names.index(i))

#print(indices)

#Clean all the documents
for i in range(0, 5):
    cleaned_doc = cleanDoc(train_1.data[i])
    train_all.data[i] = cleaned_doc

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(train_all.data)

print(count_vect.get_feature_names()[8000])


def getAllDocsForClass(train_all, index):
    doclist = []
    for i in range(0, length_total):
        if(train_all.target[i] == index):
            doclist.append(i)
    return doclist


final_word_index = []
final_terms = []
for i in range(0, len(indices)):
    print("Searching for category ,",indices[i])
    doc_class_list = getAllDocsForClass(train_all, indices[i])
    list_count = [0]* X_train_counts.shape[1]
    significant_terms = {}
    #Process a single document and increment appropriate word counts
    for j in range(0, len(doc_class_list)):
        curr_doc = doc_class_list[j]
        non_zero_word_indices = X_train_counts[curr_doc, :].nonzero()[1]
        for word_index in non_zero_word_indices:
            freq = X_train_counts[curr_doc, word_index]
            if(word_index in significant_terms):
                significant_terms[word_index] = significant_terms[word_index] + freq;
            else:
                significant_terms[word_index] = freq
            list_count[word_index] = list_count[word_index] + freq

    final_word_index.append(max(list_count))
    #For each significant term in this class, find no. of classes which contain this term
    terms = {}
    for term,freq in significant_terms.items():
        contained_classes = []
        contained_doc_numbers = X_train_counts[:, term].nonzero()[0]
        for z in contained_doc_numbers:
            class_number = train_all.target[z]
            if(class_number not in contained_classes):
                contained_classes.append(class_number)
        score = 0.5+((0.5*freq/final_word_index[i])*(math.log10(20/len(contained_classes))))
        terms[term] = score
    final_terms.append(terms)


for i in final_terms:
    map = i
    d = collections.Counter(i)
    d.most_common()
    for k, v in d.most_common(10):
        print(count_vect.get_feature_names()[k]," ", v)
    print("--------------------------------")




print(final_word_index)

        #print(non_zero_word_indices)
    #print(X_train_counts[0, 100])
    #print(X_train_counts[:,1500].nonzero()[0])
    #l1 = list(X_train_counts[0])
    #print(l1)
    #for j in range(0, len(doc_class_list)):


#Find frequency vector for the terms


#print(count_vect.get_feature_names())
#print(X_train_counts[3335,1500])

#1.Find most frequent term for each of the above classes
#idf = vectorizer.idf_
#list_of_words = vectorizer.get_feature_names()
#print(list_of_words[14000])
#print(vectors[0, 14000])




#print(dict(zip(vectorizer.get_feature_names(), idf)))
#frequency_matrix = vectorizer.fit_transform(train_1.data).astype(float)
#feature_names = np.asarray(vectorizer.get_feature_names())
#print(len(feature_names))
#print(vectors[0])
##feature_names = np.asarray(vectorizer.get_feature_names())
#feature_names = vectorizer.get_feature_names()

#print(feature_names)

#print("\n".join(train_1.data[2].split("\n")[:10]))

#print(text.ENGLISH_STOP_WORDS)


#print(list(category_count.keys()))
#Plots.histogram(list(category_count.values()), "Categories", "Frequency","Plot 1","red", 'File1', list(range(len(categories))),train_1.target_names)
# Plots.linePlot(list(category_count.keys()), list(category_count.values()),"Categories","Frequency","Plot 1","red", 'File1',list(range(len(categories))),train_1.target_names)