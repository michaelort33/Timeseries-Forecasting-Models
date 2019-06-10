import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.linear_model import LinearRegression
import math

model = load_model('../trainer/btc_predictor_5.h5')


def read(df):
    # add Date name
    df.columns = ['Date', 'price', 'volume']

    # convert to date type
    df.loc[:, 'Date'] = df.loc[:, 'Date'].astype('datetime64[ns]')

    # Use Date column as index
    df = df.set_index('Date', drop=True)

    # Use the price at the end of every hour

    return df


# create features from recent data
def create_features(df):
    chunk_size = 60
    num_chunks = math.floor(len(df) / chunk_size)

    df_features = pd.DataFrame(columns=['mean', 'std', 'slope', 'max_change', 'price'])

    for i in list(range(0, num_chunks)):
        chunk_indices = list(range(i * chunk_size, (i * chunk_size) + chunk_size))
        chunk = df.iloc[chunk_indices, :]

        x = list(range(0, chunk_size))
        lin_model = LinearRegression().fit(np.array(x).reshape(-1, 1), chunk.price.values)
        df_features.loc[i, 'slope'] = lin_model.coef_[0]
        df_features.loc[i, 'mean'] = chunk.price.mean()
        df_features.loc[i, 'std'] = chunk.price.std()
        df_features.loc[i, 'max_change'] = chunk.price.max() - chunk.price.min()

    df_features.loc[:, 'price'] = df.price[::60].values

    df_shifted_features = pd.DataFrame()
    for i in list(range(0, 4)):
        df_shifted = df_features.shift(i)
        df_shifted.columns = df_shifted.columns.values + str(i)
        df_shifted_features = pd.concat([df_shifted_features, df_shifted], axis=1)

    # remove NAs created by the shift
    df = df_shifted_features.dropna()

    return df


# create a y price value from input data that matches features
def create_y(my_price):
    data_y = my_price[::60]
    data_y = data_y[3:len(data_y)]
    data_y = data_y.shift(-1)
    data_y = data_y.dropna()

    return (data_y)


# combine features and y
def add_y_to_features(data, data_features):
    data_y = create_y(data.price)

    data_features = data_features.iloc[:-1, :]

    data_features.loc[:, 'price'] = data_y.values
    return data_features


# predict the next hour's price
def get_predictions(train_x, test_x, train_y, model=model):

    def custom_scaler(df_fit, df_scale):
        my_scaler = MinMaxScaler()

        # fit to train
        my_scaler.fit(df_fit)

        # transform train or test
        my_scaled = my_scaler.transform(df_scale)

        return my_scaled

    # scale test data
    test_x_sc = custom_scaler(train_x, test_x)

    # reshape test data to 3D for prediction
    test_x_sc_neural = test_x_sc.reshape(test_x_sc.shape[0], test_x_sc.shape[1], 1)

    # inverse scale
    def inverse_scale(df1, df2):
        my_scaler = MinMaxScaler()
        my_scaler.fit(df1)
        unscaled_predictions = my_scaler.inverse_transform(df2)

        return unscaled_predictions

    def predictions(my_test_x_sc, my_train_y, my_model):
        # predictions
        predictions_sc = my_model.predict(my_test_x_sc)

        # invert scaled predictions
        my_predictions = inverse_scale(my_train_y.values.reshape(-1, 1), predictions_sc)

        return my_predictions

    predictions_test = predictions(test_x_sc_neural, train_y, model)

    return predictions_test
