import pandas as pd
import numpy as np


WEATHER_COLS = ['humidity', 'weather_description', 'wind_direction_degree',
                'wind_speed_mph', 'degrees_kelvin']


def get_data():
    reg_data = pd.read_csv('../data/train.csv')
    weather_data = pd.read_csv('../data/NY_weather_2016.csv',
                               sep='\s*,\s*', engine='python')
    return reg_data, weather_data


def convert_datetime(i_df, func, cols):
    for col_name in cols:
        i_df[col_name] = func(i_df[col_name])
    return i_df


def round_to_hour(dt_st):
    '''Round to closet hour'''
    return pd.to_datetime(dt_st).dt.round('H')
    # return pd.to_datetime(dt_st).values.astype('<M8[m]')


def seconds_since_1970(dt_st):
    '''convert to number of seconds since 1970-ish'''
    return pd.to_datetime(dt_st).values.astype(np.int64)


def add_new_weather_column(new_column_name, column_merge_name,
                           target_df, weather_df_date_column, weather_df):
    nw_col = []
    for rw in target_df[column_merge_name]:
        nw_col.append(weather_df.loc[weather_df[weather_df_date_column] == rw,
                                     new_column_name].item())
    target_df[new_column_name] = nw_col
    return target_df


def new_weather_column(row, new_column_name, column_merge_name,
                       weather_df_date_column, weather_df):
    '''Function to create a new column given the row'''
    return weather_df.loc[
        weather_df[weather_df_date_column] == row[column_merge_name],
        new_column_name].item()


def add_weather_columns(df_t, weather_df_date_column, weather_df,
                        new_columns_names_list=WEATHER_COLS):
    '''Iterate over all the weather columns and add them to a single df'''
    for col in new_columns_names_list:
        df_t[col] = df_t.apply(
            lambda row: new_weather_column(row, col,
                                           'pickup_datetime_no_secs',
                                           'datetime', weather_df),
            axis=1)
    return df_t


def main():
    print('''I'm trying to work, leave me alone ...''')
    reg_data, weather_data = get_data()
    reg_data['pickup_datetime_no_secs'] = round_to_hour(
        reg_data['pickup_datetime'])

    reg_data_datetime_fixed = convert_datetime(
        reg_data, seconds_since_1970,
        ['pickup_datetime', 'pickup_datetime_no_secs', 'dropoff_datetime'])
    weather_data_datetime_fixed = convert_datetime(
        weather_data, seconds_since_1970, ['datetime'])

    # For testing purposes
    # reg_data_datetime_fixed = reg_data_datetime_fixed.head()

    reg_data_datetime_fixed = add_weather_columns(
        reg_data_datetime_fixed, 'pickup_datetime_no_secs',
        weather_data_datetime_fixed)

    print(reg_data_datetime_fixed.head())
    reg_data_datetime_fixed.to_csv('../data/merged_data.csv',
                                   index=False, encoding='utf-8')

    print('''I'm done, now let's go salsa dancing''')


if __name__ == '__main__':
    main()
