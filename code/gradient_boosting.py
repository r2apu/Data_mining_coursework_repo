from sklearn import ensemble, preprocessing
from linear_regression_try1 import get_error, get_data, get_some_columns


def main():
    print('working ...')
    # #############################################################################
    # Load data

    train_df, test_df = get_data()

    y_train = train_df['trip_duration']
    y_test = test_df['trip_duration']
    X_train, X_test = get_some_columns(train_df, test_df, [
        'pickup_longitude', 'dropoff_longitude', 'pickup_latitude',
        'dropoff_latitude',
        # just others
        'pickup_hour', 'pickup_minute',
        # 'pickup_month', 'pickup_day',
        # 'pickup_week', 'pickup_day_of_week',
        # weather
        # 'pickup_humidity', 'pickup_pressure',
        # 'pickup_wind_speed', 'pickup_temperature',
    ])

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # #############################################################################
    # Fit regression model
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 5,
              'learning_rate': 0.2, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X_train_sc, y_train)

    error = get_error(clf.predict(X_test_sc), y_test)
    print("The error: %.4f" % error)


if __name__ == '__main__':
    main()
