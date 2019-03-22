from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from linear_regression_try1 import get_data, get_some_columns, \
 convert_datetime, datetime_func1


def doLogisticRegression(X, X_test, y):
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    rfe = RFE(model, 16)
    rfe.fit(X[:10000], y[:10000])

    print('Logistic Regression support: {}'.format(rfe.support_))
    print('Logistic Regression ranking: {}'.format(rfe.ranking_))

    return rfe.predict(X_test[:10000])


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

    print('Doing logistic regressions')
    print(chosen_features)
    pred = doLogisticRegression(my_train_data, my_test_data, y)

    print('Logistic regression: {}'.format(
        mean_squared_error(pred, test_df['trip_duration'][:10000])))


if __name__ == '__main__':
    main()
