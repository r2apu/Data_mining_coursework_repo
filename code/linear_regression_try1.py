from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


def get_data():
    return pd.read_csv(
        '../data/export_train.csv'), pd.read_csv('../data/export_test.csv')


def get_train_test():
    data_df = get_data()
    # randomize rows of data
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    return train_test_split(data_df, test_size=0.2)


def get_squared_error(target_df, prediction_df):
    if (len(target_df) != len(prediction_df)):
        print('Error. Inputs do not have same size')
        return

    sqr_err = 0
    for el1, el2 in zip(target_df, prediction_df):
        sqr_err += (el1 - el2)**2


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


def convert_datetime(i_df, func, cols=['pickup_datetime', 'dropoff_datetime']):
    for col_name in cols:
        i_df[col_name] = func(i_df[col_name])
    return i_df


def datetime_func1(dt_st):
    '''convert to number of seconds since 1970-ish'''
    return pd.to_datetime(dt_st).values.astype(np.int64)


def datetime_func2(dt_st):
    '''convert to hour of day'''
    datetime_col = pd.to_datetime(dt_st)
    return datetime_col.dt.hour + datetime_col.dt.minute/60


def print_predictions(predictions, target):
    for el1, el2 in zip(predictions, target):
        print('{}\t{}'.format(el1[0], el2))


def main():
    print('wEllo Horld')

    train_df, test_df = get_data()

    y = train_df['trip_duration']
    # ylog = np.log(y.values+1)

    my_train_data_1, my_test_data_1 = get_some_columns(train_df, test_df)
    my_train_data_2, my_test_data_2 = get_some_columns(train_df, test_df, [
        'pickup_longitude', 'dropoff_longitude', 'pickup_latitude',
        'dropoff_latitude', 'vendor_id', 'passenger_count'])

    train_df_dt1 = convert_datetime(train_df, datetime_func1)
    test_df_dt1 = convert_datetime(test_df, datetime_func1)
    my_train_data_3, my_test_data_3 = \
        get_some_columns(train_df_dt1, test_df_dt1, [
            'pickup_longitude', 'dropoff_longitude', 'pickup_latitude',
            'dropoff_latitude', 'vendor_id', 'passenger_count',
            'pickup_datetime', 'dropoff_datetime'])

    my_train_data_4, my_test_data_4 = \
        get_some_columns(train_df_dt1, test_df_dt1, [
            'pickup_longitude', 'dropoff_longitude', 'pickup_latitude',
            'dropoff_latitude', 'vendor_id', 'passenger_count',
            'pickup_datetime', 'dropoff_datetime', 'humidity_pickup',
            'pressure_pickup', 'wind_direction_pickup',
            'wind_speed_pickup', 'temperature_pickup'
        ])

    my_train_data_4, my_test_data_4 = \
        get_some_columns(train_df_dt1, test_df_dt1, [
            'pickup_longitude', 'dropoff_longitude', 'pickup_latitude',
            'dropoff_latitude', 'vendor_id', 'passenger_count',
            'pickup_datetime', 'dropoff_datetime', 'humidity_pickup',
            'pressure_pickup', 'wind_direction_pickup',
            'wind_speed_pickup', 'temperature_pickup'
        ])

    my_train_data_5, my_test_data_5 = \
        get_some_columns(train_df_dt1, test_df_dt1, [
            'pickup_longitude', 'dropoff_longitude', 'pickup_latitude',
            'dropoff_latitude', 'vendor_id', 'passenger_count',
            'pickup_datetime', 'dropoff_datetime', 'humidity_pickup',
            'pressure_pickup', 'wind_direction_pickup',
            'wind_speed_pickup', 'temperature_pickup',
            'hour_pickup', 'minute_pickup',
            'hour_dropoff', 'minute_dropoff'
        ])

    my_train_data_6, my_test_data_6 = \
        get_some_columns(train_df_dt1, test_df_dt1, [
            'pickup_longitude', 'dropoff_longitude', 'pickup_latitude',
            'dropoff_latitude', 'vendor_id', 'passenger_count',
            'pickup_datetime', 'dropoff_datetime', 'humidity_pickup',
            'pressure_pickup', 'wind_direction_pickup',
            'wind_speed_pickup', 'temperature_pickup',
            'hour_pickup', 'minute_pickup',
            'hour_dropoff', 'minute_dropoff',
            'weekday_pickup', 'is_weekend_pickup'
        ])

    prediction1 = do_linear_regression(my_train_data_1, my_test_data_1, y)
    prediction2 = do_linear_regression(my_train_data_2, my_test_data_2, y)
    prediction3 = do_linear_regression(my_train_data_3, my_test_data_3, y)
    prediction4 = do_linear_regression(my_train_data_4, my_test_data_4, y)
    prediction5 = do_linear_regression(my_train_data_5, my_test_data_5, y)
    prediction6 = do_linear_regression(my_train_data_6, my_test_data_6, y)

    print('Just pick up and dropoff locations: {}'.format(
        mean_squared_error(prediction1, test_df['trip_duration'])))
    print(
        'Pick up, dropff locations, vendor_id, and passenger_count: {}'.format(
            mean_squared_error(prediction2, test_df['trip_duration'])))
    print(
        'Pick up, dropoff location AND time -converted to seconds since 1970-,'
        'vendor_id, passenger_count: {}'.format(
            mean_squared_error(prediction3, test_df['trip_duration'])))

    print('Adding weather: {}'.format(
        mean_squared_error(
            prediction4, test_df['trip_duration'])))

    print('Adding hour and minute pickup-dropoff: {}'.format(
        mean_squared_error(
            prediction5, test_df['trip_duration'])))

    print('plus weekend pickup and is_weekend_pickup: {}'.format(
        mean_squared_error(
            prediction6, test_df['trip_duration'])))


if __name__ == '__main__':
    main()
