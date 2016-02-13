from sklearn.datasets import fetch_20newsgroups
import Plots

categories = ['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey']

train_1 = fetch_20newsgroups(subset='train', categories = categories, shuffle = True, random_state=42)

#print("\n".join(train_1.data[0].split("\n")[:3]))

length = len(train_1.data)
print("Total records = ",len(train_1.data))

print(len(train_1))
print(train_1.target_names)

tick_list = list(range(len(categories)))
print(tick_list)
category_count = {}

for i in range(0,length):
    category = train_1.target_names[train_1.target[i]]
    if(category in category_count):
        counter = category_count[category]
        counter = counter + 1
    else:
        counter = 1
    category_count[category] = counter


Plots.barPlot(category_count)


#print(list(category_count.keys()))
#Plots.histogram(list(category_count.values()), "Categories", "Frequency","Plot 1","red", 'File1', list(range(len(categories))),train_1.target_names)
# Plots.linePlot(list(category_count.keys()), list(category_count.values()),"Categories","Frequency","Plot 1","red", 'File1',list(range(len(categories))),train_1.target_names)