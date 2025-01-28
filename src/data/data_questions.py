import json
import pickle

from pandas import read_csv, DataFrame, Series, to_datetime, to_numeric, concat


def expand_user_data(row: Series) -> DataFrame:
    user_id = row['user_id']
    countries = eval(row['country'])
    Rs = eval(row['R'])
    Fs = eval(row['F'])
    Ms = eval(row['M'])

    return DataFrame({'user_id': [user_id]*len(countries), 'country': countries, 'R': Rs, 'F': Fs, 'M': Ms})


def question_1(products_df: DataFrame) -> dict:
    """
    Q1: - Which product (partnumber) with color_id equal to 3 belongs to the lowest family code with a discount?
    """

    result = products_df[(products_df['color_id'] == 3) & (products_df['discount'] == 1)] \
        .sort_values('family').iloc[0]['partnumber']

    return {'partnumber': int(result)}


def question_2(user_df: DataFrame) -> dict:
    """
    Q2: - In the country where most users have made purchases totaling less than 500 (M) , which is the user who has the
    lowest purchase frequency (F), the most recent purchase (highest R) and the lowest user_id? Follow the given order
    of variables as the sorting priority.
    """

    user_int_df = user_df.astype({'user_id': 'int'})
    filtered = user_int_df[user_int_df['M'] < 500]

    country_counts = filtered.groupby('country')['user_id'].nunique().reset_index(name='user_count')

    max_count = country_counts['user_count'].max()
    target_country = country_counts[country_counts['user_count'] == max_count].iloc[0]['country']

    target_users = filtered[filtered['country'] == target_country]

    target_users_sorted = target_users.sort_values(by=['F', 'R', 'user_id'], ascending=[True, False, True])

    top_user = target_users_sorted.iloc[0]

    return {'user_id': int(top_user['user_id'])}


def question_3(train_df: DataFrame) -> dict:
    """
    Q3: - Among the products that were added to the cart at least once, how many times is a product visited
    before it is added to the cart on average? Give the answer with 2 decimals.
    """

    train_df['timestamp_local'] = to_datetime(train_df['timestamp_local'])
    train_df = train_df.sort_values(by=['session_id', 'partnumber', 'timestamp_local'])

    first_add_times = (
        train_df[train_df['add_to_cart'] == 1]
        .groupby(['session_id', 'partnumber'])['timestamp_local']
        .min()
        .reset_index(name='first_add_time')
    )

    merged_df = train_df.merge(first_add_times, on=['session_id', 'partnumber'], how='inner')

    visits_before_add = merged_df[
        (merged_df['add_to_cart'] == 0) &
        (merged_df['timestamp_local'] < merged_df['first_add_time'])
        ].groupby(['session_id', 'partnumber']).size()

    average_visits = visits_before_add.mean() if not visits_before_add.empty else 0.0
    average_visits = float(round(average_visits, 2))

    return {"average_previous_visits": average_visits}


def question_4(train_df: DataFrame, products_df: DataFrame) -> dict:
    """
    Q4: - Which device (device_type) is most frequently used by users to make purchases (add_to_cart = 1) of discounted
    products (discount = 1)?
    """

    is_discounted_df = products_df[['partnumber', 'discount']]

    train_df = train_df.merge(is_discounted_df, on='partnumber', how='inner')

    train_df['add_to_cart'] = to_numeric(train_df['add_to_cart'], errors='coerce').fillna(0)
    train_df['discount'] = to_numeric(train_df['discount'], errors='coerce').fillna(0)

    filtered_df = train_df[(train_df['add_to_cart'] == 1) & (train_df['discount'] == 1)]

    counts = filtered_df.groupby('device_type').size()

    if len(counts) == 0:
        return {"device_type": None}

    most_frequent_device = int(counts.idxmax())

    return {"device_type": most_frequent_device}


def question_5(user_df: DataFrame, users_sessions_df: DataFrame) -> dict:
    """
    Q5: - Among users with purchase frequency (F) in the top 3 within their purchase country, who has interacted with
    the most products (partnumber) in sessions conducted from a device with identifier 3 (device_type = 3)?
    """

    sorted_user_df = user_df.sort_values(['country', 'F'], ascending=[True, False])
    top_users_by_country = sorted_user_df.groupby('country').head(3)['user_id'].unique()

    filtered_sessions_df = users_sessions_df[
        (users_sessions_df['user_id'].isin(top_users_by_country)) & (users_sessions_df['device_type'] == 3)]

    product_counts_df = filtered_sessions_df.groupby('user_id')['partnumber'].nunique().reset_index(
        name='distinct_products')

    if product_counts_df.empty:
        return {"user_id": None, "distinct_products": 0}

    return {"user_id": int(product_counts_df.loc[product_counts_df['distinct_products'].idxmax()].user_id)}


def question_6(user_df: DataFrame, users_sessions_df: DataFrame, products_df: DataFrame) -> dict:
    """
    Q6: - For interactions that occurred outside the user's country of residence, how many unique family identifiers are
    there?
    """

    user_countries = user_df.groupby('user_id')['country'].agg(lambda x: set(x)).reset_index(name='residence_countries')

    sessions_merged = users_sessions_df.merge(products_df[['partnumber', 'family']], on='partnumber', how='left')

    sessions_merged = sessions_merged.merge(user_countries, on='user_id', how='left')

    outside_interactions = sessions_merged[
        sessions_merged['residence_countries'].notnull() &
        sessions_merged['country'].notnull() &
        ~sessions_merged.apply(lambda row: row['country'] in row['residence_countries'], axis=1)
        ]

    return {"unique_families": outside_interactions['family'].nunique()}


def question_7(users_sessions_df: DataFrame, products_df: DataFrame) -> dict:
    """
    Q7: - Among interactions from the first 7 days of June, which is the most frequent page type where each family is
    added to the cart? Return it in the following format: {'('family'): int('most_frequent_pagetype')} . In case of a
    tie, return the smallest pagetype.
    """

    mask_date = (users_sessions_df['date'] >= '2024-06-01') & (users_sessions_df['date'] <= '2024-06-07')
    june_sessions_df = users_sessions_df[mask_date]

    target_sessions_df = june_sessions_df[june_sessions_df['add_to_cart'] == 1]

    target_sessions_df = target_sessions_df.merge(products_df[['partnumber', 'family']], on='partnumber', how='left')

    pagetype_by_family = target_sessions_df.groupby(['family', 'pagetype']).size().reset_index(name='count')
    pagetype_by_family = pagetype_by_family \
        .sort_values(by=['family', 'count', 'pagetype'], ascending=[True, False, True])

    most_freq_pagetype_by_family = pagetype_by_family \
        .groupby('family').head(1).astype({'family': 'str', 'pagetype': 'int'})

    return dict(zip(most_freq_pagetype_by_family['family'], most_freq_pagetype_by_family['pagetype']))


if __name__ == "__main__":
    user_df = read_csv('data/raw/user_data.csv')
    train_df = read_csv('data/raw/train.csv')

    with open('data/raw/products.pkl', 'rb') as f:
        products_df = pickle.load(f)

    expanded_user_df = user_df.apply(expand_user_data, axis=1)
    expanded_user_df = concat(expanded_user_df.tolist(), ignore_index=True)
    expanded_user_df = expanded_user_df.astype({'user_id': 'str'})

    users_sessions_df = train_df.dropna(subset=['user_id'])
    users_sessions_df = users_sessions_df.astype({'user_id': 'int'}).astype({'user_id': 'str'})

    result_question_1 = question_1(products_df)
    result_question_2 = question_2(expanded_user_df)
    result_question_3 = question_3(train_df)
    result_question_4 = question_4(train_df, products_df)
    result_question_5 = question_5(expanded_user_df, users_sessions_df)
    result_question_6 = question_6(expanded_user_df, users_sessions_df, products_df)
    result_question_7 = question_7(train_df, products_df)

    final_result = {
        'target': {
            'query_1': result_question_1,
            'query_2': result_question_2,
            'query_3': result_question_3,
            'query_4': result_question_4,
            'query_5': result_question_5,
            'query_6': result_question_6,
            'query_7': result_question_7,
        }
    }
    
    with open('predictions/predictions_1.json', 'w') as archivo:
        json.dump(final_result, archivo, indent=4)
