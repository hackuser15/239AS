import pandas
from sklearn.feature_extraction import DictVectorizer

def one_hot_dataframe(data, cols, replace=False):
    """ Takes a dataframe and a list of columns that need to be encoded.
        Returns a 3-tuple comprising the data, the vectorized data,
        and the fitted vectorizor.
    """
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pandas.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)

training_data = pandas.read_csv('D:/WINTER/EE239/Project1/network_backup_dataset.csv')
#for row in training_data:
#    pprint(row)

df, _, _ = one_hot_dataframe(training_data, ['Day of Week', 'Work-Flow-ID','File Name'], replace=True)
print(df)