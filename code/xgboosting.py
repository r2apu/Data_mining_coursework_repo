import xgboost as xgb
from linear_regression_try1 import get_error, get_data, get_some_columns


def main():
    print('working ...')

    train_df, test_df = get_data()

    y_train = train_df['trip_duration']
    y_test = test_df['trip_duration']
    X_train, X_test = get_some_columns(train_df, test_df, [
        'pickup_longitude', 'dropoff_longitude', 'pickup_latitude',
        'dropoff_latitude',
        # just others
        # 'pickup_hour', 'pickup_minute',
        # 'pickup_month', 'pickup_day',
        # 'pickup_week', 'pickup_day_of_week',
        # weather
        # 'pickup_humidity',
        # 'pickup_pressure',
        # 'pickup_wind_speed', 'pickup_temperature',
    ])

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    # #############################################################################
    # Fit regression model
    params = {'max_depth': 12, 'eta': 0.5}
    bst = xgb.train(params, dtrain, 10)

    # print(bst.predict(dtest))
    # print(y_test)

    error = get_error(bst.predict(dtest), y_test)
    print("The error: %.4f" % error)


if __name__ == '__main__':
    main()
