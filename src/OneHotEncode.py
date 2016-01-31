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

# def transform_day_to_number(data, cols):
#     for col in cols:
#         data[col] = data.apply(lambda row: day_to_number(row[col]), axis=1)
#     return data
#
# def transform_number_from_end(data, cols):
#     for col in cols:
#         data[col] = data.apply(lambda row: extract_number_from_end(row[col]), axis=1)
#     return data
#
# def day_to_number(str):
#     map = {'Monday': 0.0, 'Tuesday': 1.0, 'Wednesday': 2.0, 'Thursday': 3.0, 'Friday': 4.0, 'Saturday': 5.0, 'Sunday': 6.0}
#     return map[str]
#
# def extract_number_from_end(str):
#     return [float(s) for s in str.split('_') if s.isdigit()][-1]