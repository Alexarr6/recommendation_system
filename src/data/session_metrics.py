import pandas as pd


def get_session_metrics(df: pd.DataFrame, user_id: int) -> pd.DataFrame:
    """
    Given a pandas DataFrame in the format of the train dataset and a user_id, return the following metrics for every
    session_id of the user:
        - user_id (int) : the given user id.
        - session_id (int) : the session id.
        - total_session_time (float) : The time passed between the first and last interactions, in seconds.
            Rounded to the 2nd decimal.
        - cart_addition_ratio (float) : Percentage of the added products out of the total products interacted with.
            Rounded ot the 2nd decimal.

    If there's no data for the given user, return an empty Dataframe preserving the expected columns.
    The column order and types must be strictly followed.

    Parameters
    ----------
    df : pandas DataFrame
       DataFrame  of the data to be used for the agent.
    user_id : int
        ID of the client.

    Returns
    -------
    Pandas Dataframe with some metrics for all the sessions of the given user.
    """

    df = df.dropna(subset=['user_id', 'session_id'])
    df = df.astype({'user_id': 'int', 'session_id': 'int'})
    df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])

    user_transactions_df = df[df['user_id'] == user_id]

    if user_transactions_df.empty:
        return pd.DataFrame(columns=['user_id', 'session_id', 'total_session_time', 'cart_addition_ratio'])

    user_session_duration_df = user_transactions_df. \
        groupby(['user_id', 'session_id'])['timestamp_local'].agg(['min', 'max'])
    user_session_duration_df['total_session_time'] = (
            user_session_duration_df['max'] - user_session_duration_df['min']
    ).dt.total_seconds()

    user_session_duration_df['total_session_time'] = user_session_duration_df['total_session_time'].round(2)
    user_session_duration_df = user_session_duration_df.reset_index()

    user_cart_avg_session_df = user_transactions_df. \
        groupby(['user_id', 'session_id'])['add_to_cart'].mean().reset_index()
    user_cart_avg_session_df['cart_addition_ratio'] = (user_cart_avg_session_df['add_to_cart'] * 100).round(2)

    result_df = user_session_duration_df.merge(user_cart_avg_session_df, on=['user_id', 'session_id'], how='inner')

    result_df = result_df[['user_id', 'session_id', 'total_session_time', 'cart_addition_ratio']]
    result_df = result_df.sort_values(by=['user_id', 'session_id']).reset_index(drop=True)

    return result_df


if __name__ == "__main__":

    train_df = pd.read_csv('data/raw/train.csv')
    user_session_duration_df = get_session_metrics(train_df, 2)
