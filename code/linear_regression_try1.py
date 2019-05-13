from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn import preprocessing
import numpy as np
import math


def rmsle(p, a):
    # return math.sqrt(mean_squared_log_error(np.absolute(p), np.absolute(a)))
    return math.sqrt(mean_squared_log_error(np.absolute(p), a))


def get_data():
    # '../data/export_train.csv'), pd.read_csv('../data/export_test.csv')
    return pd.read_csv('../data/train_merged_datetime.csv'), pd.read_csv(
        '../data/test_merged_datetime.csv')


def get_train_test():
    data_df = get_data()
    # randomize rows of data
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    return train_test_split(data_df, test_size=0.2)


def do_linear_regression(X_train, X_test, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y)
    return lin_reg.predict(X_test)


def get_some_columns(train_df, test_df,
                     columns=['pickup_longitude', 'dropoff_longitude',
                              'pickup_latitude', 'dropoff_latitude']):
    train_data = train_df[columns]
    test_data = test_df[columns]
    return train_data, test_data


def convert_datetime(i_df, func, cols=['pickup_datetime']):
    for col_name in cols:
        i_df[col_name] = func(i_df[col_name])
    return i_df


def datetime_func1(dt_st):
    '''convert to number of seconds since 1970-ish'''
    return pd.to_datetime(dt_st).values.astype(np.int64)


def print_predictions(predictions, target):
    for el1, el2 in zip(predictions, target):
        print('{}\t{}'.format(el1[0], el2))


def get_error(pred, act, method=mean_squared_error):
    method = rmsle
    return method(pred, act)


def main():
    print('wEllo Horld')

    train_df, test_df = get_data()
    y = train_df['trip_duration'].reshape(-1, 1)
    y_test = test_df['trip_duration'].reshape(-1, 1)
    # ylog = np.log(y.values+1)

    my_train_data_1, my_test_data_1 = get_some_columns(train_df, test_df)

    my_train_data_2, my_test_data_2 = get_some_columns(train_df, test_df, [
        'pickup_longitude', 'dropoff_longitude', 'pickup_latitude',
        'dropoff_latitude',
        # just others
        'pickup_hour', 'pickup_minute',
    ])

    my_train_data_3, my_test_data_3 = get_some_columns(train_df, test_df, [
        'pickup_longitude', 'dropoff_longitude', 'pickup_latitude',
        'dropoff_latitude',
        # just others
        'pickup_hour', 'pickup_minute',
        # weather
        'pickup_humidity',
        'pickup_pressure', 'pickup_wind_speed',
        'pickup_wind_speed', 'pickup_temperature',
    ])

    my_train_data_4, my_test_data_4 = get_some_columns(train_df, test_df, [
        'pickup_longitude', 'dropoff_longitude', 'pickup_latitude',
        'dropoff_latitude',
        # just others
        'pickup_hour', 'pickup_minute',
        # weather
        'pickup_humidity',
        'pickup_pressure', 'pickup_wind_speed',
        'pickup_wind_speed', 'pickup_temperature',
    ])

    scaler_X = preprocessing.StandardScaler().fit(my_train_data_4)
    scaler_y = preprocessing.StandardScaler().fit(y)

    X_train_sc = scaler_X.transform(my_train_data_4)
    X_test_sc = scaler_X.transform(my_test_data_4)

    y_train_sc = scaler_y.transform(y)
    y_test_sc = scaler_y.transform(y_test)

    predictionScale = do_linear_regression(X_train_sc, X_test_sc, y_train_sc)

    prediction1 = do_linear_regression(my_train_data_1, my_test_data_1, y)
    prediction2 = do_linear_regression(my_train_data_2, my_test_data_2, y)
    prediction3 = do_linear_regression(my_train_data_3, my_test_data_3, y)
    prediction4 = do_linear_regression(my_train_data_4, my_test_data_4, y)

    print('Just pick up and dropoff locations: {}'.format(
        get_error(prediction1, test_df['trip_duration'])))

    print('Just pick up and dropoff locations plus other stuff: {}'.format(
        get_error(prediction2, test_df['trip_duration'])))

    print('Adding weather: {}'.format(
        get_error(prediction3, test_df['trip_duration'])))

    print('Adding weekend: {}'.format(
        get_error(prediction4, test_df['trip_duration'])))

    print('Scaled vals: {}'.format(get_error(
        scaler_y.inverse_transform(predictionScale), y_test)))


if __name__ == '__main__':
    main()
