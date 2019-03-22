from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesClassifier
from linear_regression_try1 import get_data, get_some_columns, \
 convert_datetime, datetime_func1


def doExtraTreeClassifier(X, X_test, y):
    model = ExtraTreesClassifier(n_estimators=2)
    model.fit(X[:1000], y[:1000])
    return model.predict(X_test[:1000]), model.feature_importances_


def main():
    print('De la cana se hace el guaro')

    train_df, test_df = get_data()
    y = train_df['trip_duration']

    train_df_dt = convert_datetime(train_df, datetime_func1)
    test_df_dt = convert_datetime(test_df, datetime_func1)
    chosen_features = [
        'pickup_longitude', 'dropoff_longitude', 'pickup_latitude',
        'dropoff_latitude', 'vendor_id', 'passenger_count',
        'pickup_datetime', 'dropoff_datetime', 'humidity_pickup',
        'pressure_pickup', 'wind_direction_pickup',
        'wind_speed_pickup', 'temperature_pickup',
        'hour_pickup', 'minute_pickup',
        'hour_dropoff', 'minute_dropoff',
        'weekday_pickup', 'is_weekend_pickup'
    ]
    my_train_data, my_test_data = get_some_columns(
        train_df_dt, test_df_dt, chosen_features)

    print('Doing tree classifier ...')
    pred, weights = doExtraTreeClassifier(my_train_data, my_test_data, y)
    for it, w in enumerate(weights):
        print('{}: {}'.format(chosen_features[it], w))

    print('Tree classifier: {}'.format(
        mean_squared_error(pred, test_df['trip_duration'][:1000])))


if __name__ == '__main__':
    main()
