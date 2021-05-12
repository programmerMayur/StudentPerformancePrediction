import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle


def data_split(data,ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)*ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

if __name__ == "__main__":
    df = pd.read_csv('data/data1.csv')
    df = df.round(2)
    #print(df.head())

    train, test = data_split(df, 0.2)

    x_train = train[['One','Two','Three','Four','Five','Six','Seven']].to_numpy()
    x_test = test[['One','Two','Three','Four','Five','Six','Seven']].to_numpy()

    y_train = train[['Final']].to_numpy().reshape(1200,-1)
    y_test = test[['Final']].to_numpy().reshape(300 ,-1)

    regi = linear_model.LinearRegression()

    reg = regi.fit(x_train, y_train)

    file = open('pkl/model1.pkl','wb')
    pickle.dump(reg, file)
    file.close()
