import numpy as np
import pandas as pd


def check_unique_values(data):
    if data.ndim == 1:
        print(np.unique(data).size)
    else:
        for i in data:
            print(i)
            print(np.unique(data[i]).size)


def one_hot_encode(raw_data, upper_limit=20,lower_limit=0, debug=False):
    new_data = pd.DataFrame({})
    for data in raw_data:
        col_type = raw_data[data].dtype
        col = raw_data[data]
        if col_type == 'object' or lower_limit < col.nunique() < upper_limit:
            one_hot = pd.get_dummies(col, prefix=data)
            new_data = pd.concat([new_data, one_hot], axis=1)
        else:
            new_data = pd.concat([new_data, col], axis=1)
    if debug:
        new_data.info()
        print(new_data.shape)
    return new_data


if __name__ == '__main__':
    from load_csv import DataSet
    data = DataSet()
    check_unique_values(data.get_trainY())
    one_hot_encode(data.get_trainX_pd(), debug=True)
