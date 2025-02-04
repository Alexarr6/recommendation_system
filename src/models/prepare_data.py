import pickle
from typing import List, Dict, Tuple
import json

import numpy as np
from pandas import DataFrame, Series, to_datetime, api, notna, concat, read_csv, Timestamp, merge
from numpy import random
from sklearn.preprocessing import MinMaxScaler


class DataManager:
    """
    Responsible for loading data and expanding rows.
    """

    @staticmethod
    def expand_row(row: Series) -> DataFrame:
        """
        Example method that expands a single row, duplicating user_id and splitting columns
        like 'R', 'F', and 'M' across multiple rows.
        """
        user_id = row['user_id']
        countries = eval(row['country'])
        Rs = eval(row['R'])
        Fs = eval(row['F'])
        Ms = eval(row['M'])

        data = {
            'user_id': [user_id] * len(countries),
            'country': countries,
            'R': Rs,
            'F': Fs,
            'M': Ms
        }
        return DataFrame(data)

    @staticmethod
    def expand_dataframe(user_df: DataFrame) -> DataFrame:
        """
        Applies expand_row to an entire user DataFrame.
        """
        expanded_dfs = user_df.apply(DataManager.expand_row, axis=1)
        user_df = concat(expanded_dfs.tolist(), ignore_index=True)
        user_df = user_df.astype({'user_id': 'str', 'country': 'str'})
        return user_df


class Preprocessor:
    """
    Responsible for data cleaning and preparation, including time calculations.
    """

    def preprocess_data(self, product_df: DataFrame, sessions_df: DataFrame, user_df: DataFrame) -> DataFrame:
        """
        Merges product, session, and user data, then scales R, F, M columns.
        """

        sessions_df['user_id'] = sessions_df['user_id'].apply(lambda x: str(int(x)) if notna(x) else 'Unknown')
        sessions_df = sessions_df.astype({'country': 'str'})
        sessions_df['timestamp_local'] = to_datetime(sessions_df['timestamp_local'], errors='coerce')
        sessions_df['date'] = to_datetime(sessions_df['date'], errors='coerce')

        sessions_df = self._calculate_time_between_products(sessions_df)

        sessions_merged = sessions_df.merge(
            product_df[['partnumber', 'family', 'cod_section', 'color_id', 'discount']],
            on='partnumber',
            how='left'
        )
        sessions_merged = sessions_merged.merge(
            user_df[['user_id', 'country', 'R', 'F', 'M']],
            on=['user_id', 'country'],
            how='left'
        )

        sessions_merged = sessions_merged.dropna(subset=['partnumber'])

        scaler = MinMaxScaler()
        sessions_merged[['R', 'F', 'M']] = scaler.fit_transform(sessions_merged[['R', 'F', 'M']].fillna(0))

        return sessions_merged

    @staticmethod
    def delete_long_sessions(train_df: DataFrame) -> DataFrame:

        session_length_df = train_df.groupby('session_id').size().reset_index(name='count')
        valid_sessions_id = set(session_length_df[(session_length_df['count'] < 100)]['session_id'].tolist())
        train_df = train_df[train_df['session_id'].isin(valid_sessions_id)]

        return train_df

    @staticmethod
    def delete_short_sessions(train_df: DataFrame) -> DataFrame:

        session_length_df = train_df.groupby('session_id').size().reset_index(name='count')
        valid_sessions_id = set(session_length_df[(session_length_df['count'] > 3)]['session_id'].tolist())
        train_df = train_df[train_df['session_id'].isin(valid_sessions_id)]

        return train_df

    @staticmethod
    def remove_mistake_clicks(train_df: DataFrame) -> DataFrame:
        train_df = train_df[(train_df['time_to_next_seconds'] >= 2) | (train_df['add_to_cart'] > 0)]

        return train_df

    @staticmethod
    def _calculate_time_between_products(df: DataFrame, section: int = None) -> DataFrame:
        """
        Sorts the data by session_id and timestamp, then calculates time deltas
        to the next product in seconds. Optionally, filters by a given section.
        """
        if not api.types.is_datetime64_any_dtype(df['timestamp_local']):
            df['timestamp_local'] = to_datetime(df['timestamp_local'])

        if section is not None and 'cod_section' in df.columns:
            df = df[df['cod_section'] == section]

        df = df.sort_values(by=['session_id', 'timestamp_local'])
        df['time_to_next'] = df.groupby('session_id')['timestamp_local'].shift(-1) - df['timestamp_local']
        df['time_to_next_seconds'] = df['time_to_next'].dt.total_seconds()
        df.drop('time_to_next', axis=1, inplace=True)

        return df

    @staticmethod
    def create_evaluation_subset(train_df: DataFrame, n_sessions: int = 5000, random_seed: int = 1) -> DataFrame:
        """
        1) Loads the full training data (which has add_to_cart).
        2) Randomly picks n_sessions sessions.
        """
        random.seed(random_seed)

        train_with_buys_df = train_df.copy()
        train_with_buys_df = train_with_buys_df[train_with_buys_df['add_to_cart'] > 0]
        buys_per_session_df = train_with_buys_df.groupby(['session_id']).size().reset_index(name='n_buys')
        buys_per_session_df = buys_per_session_df[
            (buys_per_session_df['n_buys'] >= 5) & (buys_per_session_df['n_buys'] <= 10)]

        unique_sessions = buys_per_session_df['session_id'].drop_duplicates()
        chosen_sessions = unique_sessions.sample(n_sessions, random_state=random_seed)
        one_row_sessions = random.choice(chosen_sessions, size=int(len(chosen_sessions) * 0.5), replace=False)

        eval_df = train_df[train_df['session_id'].isin(chosen_sessions)].copy()
        eval_buys_df = eval_df[eval_df['add_to_cart'] > 0].copy()
        eval_session_product_map = eval_buys_df.groupby('session_id')['partnumber'].apply(list).to_dict()

        one_row_sessions_df = eval_df[eval_df['session_id'].isin(one_row_sessions)].groupby('session_id').head(1)
        several_row_sessions_df = eval_df[~eval_df['session_id'].isin(one_row_sessions)]
        eval_df = concat([one_row_sessions_df, several_row_sessions_df]).sort_index()

        return eval_df, eval_session_product_map


class ProductsInformationExtractor:
    """
    Manages logic for retrieving popular products or other popularity-based utilities.
    """

    @staticmethod
    def get_popular_products(train_df: DataFrame, top_n: int = 100) -> List[str]:
        """
        Returns the top_n most frequent products (by 'partnumber') from train_df.
        """

        long_sessions_train_df = train_df[train_df['time_to_next_seconds'] >= 2].copy()
        product_counts = long_sessions_train_df['partnumber'].value_counts().head(top_n)
        return product_counts.index.tolist()

    @staticmethod
    def create_discount_map(products_df: DataFrame) -> Dict[str, False]:
        """
        Creates a dictionary mapping partnumber -> bool(discount).
        Assumes products_df has ['partnumber', 'discount'].
        """
        discount_map = {}
        for _, row in products_df.iterrows():
            product_id = int(row['partnumber'])
            discount = bool(row['discount'])
            discount_map[product_id] = discount
        return discount_map

    @staticmethod
    def build_user_bought_map(train_df: DataFrame) -> Dict[str, List[int]]:
        """
        Creates a dictionary mapping each user_id to a list of partnumbers
        that they have actually purchased (add_to_cart > 0).
        """

        purchased_df = train_df[train_df['add_to_cart'] > 0].copy()
        user_items = purchased_df.groupby('user_id')['partnumber'].agg(list).reset_index()
        user_items = user_items[user_items['user_id'] != 'Unknown']

        user_bought_map = {}
        for _, row in user_items.iterrows():
            user_str = str(row['user_id'])
            partnumbers = row['partnumber']
            user_bought_map[user_str] = partnumbers

        return user_bought_map

    @staticmethod
    def build_user_not_bought_map(train_df: DataFrame) -> Dict[str, Dict[int, dict]]:
        """
        This helps identify which products have been bought (purchased>0) vs not bought (purchased=0),
        as well as how many times they were viewed and purchased.
        """

        df = train_df[train_df['user_id'] != 'Unknown'].copy()
        df['user_id'] = df['user_id'].astype(str)

        group_df = (
            df.groupby(['user_id', 'partnumber'], as_index=False)
            .agg(views_count=('partnumber', 'size'),
                 purchased_count=('add_to_cart', 'sum'))
        )

        group_df = group_df[group_df['purchased_count'] == 0]
        group_df = group_df.sort_values(['user_id', 'views_count'], ascending=[True, False])

        user_data = {}
        partnumber_list = []
        for row in group_df.itertuples(index=False):
            user = row.user_id
            part = row.partnumber

            if user not in user_data:
                user_data[user] = {}
                partnumber_list = []

            partnumber_list.append(part)
            user_data[user] = partnumber_list

        return user_data

    @staticmethod
    def build_user_profiles(train_df: DataFrame) -> Dict[str, Dict[tuple, dict]]:
        """
        Logic:
        1) Group by (user_id, family, cod_section, partnumber).
        2) Count how many rows => "views_count"; sum(add_to_cart>0) => "purchased_count".
        3) Then regroup by (user_id, family, cod_section) to build the final structure:
           - "bought"=1 if sum of purchased_count>0 across all partnumbers
           - "visits"= sum of views_count across all partnumbers
           - "bought_products" => for each partnumber with purchased_count>0
           - "not_bought_products" => for each partnumber with purchased_count=0
        """

        train_df = train_df.copy()
        train_df['purchased_event'] = (train_df['add_to_cart'] > 0).astype(int)
        train_df = train_df.dropna(subset=['family', 'cod_section'])
        train_df = train_df[train_df['user_id'] != 'Unknown']

        grouped_products = (
            train_df.groupby(['user_id', 'family', 'cod_section', 'partnumber'], dropna=False)
            .agg(
                views_count=('partnumber', 'size'),
                purchased_count=('purchased_event', 'sum')
            )
            .reset_index()
        )

        user_profiles = {}

        temp_map = {}

        for row in grouped_products.itertuples(index=False):
            user_id = row.user_id
            fam = row.family
            cod = row.cod_section
            part = row.partnumber
            views = row.views_count
            purchased = row.purchased_count

            key = (user_id, fam, cod)
            if key not in temp_map:
                temp_map[key] = []
            temp_map[key].append({
                'partnumber': part,
                'views_count': views,
                'purchased_count': purchased
            })

        for (user_id, fam, cod), products_list in temp_map.items():

            if user_id not in user_profiles:
                user_profiles[user_id] = {}

            total_visits = 0
            any_purchase = 0
            bought_products = {}
            not_bought_products = {}

            for product_info in products_list:
                part = product_info['partnumber']
                views = product_info['views_count']
                purchased_count = product_info['purchased_count']

                total_visits += views

                if purchased_count > 0:
                    bought_products[part] = views
                    any_purchase = 1
                else:
                    not_bought_products[part] = views

            user_profiles.setdefault(user_id, {})
            user_profiles[user_id][str(int(cod)) + '-' + str(int(fam))] = {
                "bought": any_purchase,
                "visits": total_visits,
                "bought_products": bought_products,
                "not_bought_products": not_bought_products
            }

        return user_profiles

    @staticmethod
    def compute_out_of_stock_country_products(
        train_df: DataFrame,
        stock_threshold: int,
        date_threshold: str
    ) -> List[int]:

        total_sales = (
            train_df.groupby(['partnumber'], as_index=False)['add_to_cart']
            .sum()
            .rename(columns={'add_to_cart': 'total_sales'})
        )
        total_sales = total_sales[total_sales['total_sales'] > stock_threshold]
        target_products = total_sales['partnumber'].tolist()

        max_date_df = train_df.groupby(['partnumber'], as_index=False)['timestamp_local'].max().rename(
            columns={'timestamp_local': 'max_timestamp'})
        max_date_df = max_date_df[max_date_df['partnumber'].isin(target_products)]
        max_date_df = max_date_df[max_date_df['max_timestamp'] < date_threshold]

        return max_date_df['partnumber'].tolist()

    def compute_out_of_stock_products(
        self,
        train_df: DataFrame,
        stock_threshold: int,
        date_threshold: str
    ) -> Dict[str, Dict[str, str]]:

        out_of_stock = {}
        for country in ['25', '29', '34', '57']:
            country_train_df = train_df[train_df['country'] == country]
            out_of_stock_products = self.compute_out_of_stock_country_products(
                country_train_df, stock_threshold, date_threshold
            )
            out_of_stock[country] = out_of_stock_products

        return out_of_stock

    @staticmethod
    def build_section_family_products(train_df: DataFrame) -> Dict[str, dict]:
        """
        Steps:
          1) Group by (cod_section, family, partnumber) -> count how many rows => 'views_count'
          2) Group by (cod_section, family, partnumber) with add_to_cart>0 => 'bought_count'
          3) For each (cod_section, family), pick top 5 by views_count, top 5 by bought_count
          4) Build final dictionary
        """
        train_df = train_df[train_df['timestamp_local'] >= '2024-06-10'].copy()

        train_df = train_df[~train_df['cod_section'].isna()]
        train_df['cod_section'] = train_df['cod_section'].apply(lambda x: int(x))

        views_df = train_df.groupby(['cod_section', 'family', 'partnumber']).size().reset_index(name='views_count')

        df_bought = train_df.copy()
        df_bought['bought_event'] = (df_bought['add_to_cart'] > 0).astype(int)

        bought_df = df_bought.groupby(['cod_section', 'family', 'partnumber'])['bought_event']. \
            sum().reset_index(name='bought_count')

        result_map = {}

        for (section, fam), group in views_df.groupby(['cod_section', 'family']):
            top_viewed = group.sort_values('views_count', ascending=False).head(5)
            viewed_list = list(zip(top_viewed['partnumber'], top_viewed['views_count']))

            result_map[str(section) + '_' + str(fam)] = {"top_viewed": viewed_list, "top_bought": []}

        for (section, fam), group in bought_df.groupby(['cod_section', 'family']):
            top_bought = group.sort_values('bought_count', ascending=False).head(5)
            bought_list = list(zip(top_bought['partnumber'], top_bought['bought_count']))

            if str(section) + '_' + str(fam) not in result_map:
                result_map[str(section) + '_' + str(fam)] = {"top_viewed": [], "top_bought": bought_list}
            else:
                result_map[str(section) + '_' + str(fam)]["top_bought"] = bought_list

        return result_map

    @staticmethod
    def compute_discount_ratio(session_df: DataFrame, products_df: DataFrame) -> Tuple[DataFrame, DataFrame]:

        session_df = session_df[['session_id', 'partnumber']].merge(
            products_df[['partnumber', 'discount']], on='partnumber', how='inner'
        ).drop_duplicates()
        session_length_df = session_df.groupby('session_id').size().reset_index(name='session_length')
        discount_length_df = session_df.groupby(['session_id', 'discount']).size().reset_index(name='discount_length')
        discount_length_df = discount_length_df[discount_length_df['discount'] == 1]
        session_df = session_df.merge(session_length_df, on='session_id', how='inner')
        discount_sessions_df = session_df.merge(discount_length_df, on='session_id', how='inner')
        discount_sessions_df = discount_sessions_df[
            ['session_id', 'session_length', 'discount_length']].drop_duplicates()
        discount_sessions_df['discount_ratio'] = discount_sessions_df['discount_length'] / discount_sessions_df[
            'session_length']

        return discount_sessions_df, discount_length_df

    @staticmethod
    def compute_sessions_types(
        discount_sessions_df: DataFrame,
        discount_length_df: DataFrame,
        session_df: DataFrame
    ) -> DataFrame:

        full_discount_sessions_list = discount_length_df['session_id'].tolist()
        new_season_sessions_list = list(
            set(session_df[~session_df['session_id'].isin(full_discount_sessions_list)]['session_id'].tolist()))
        sales_sessions_list = list(
            set(discount_sessions_df[discount_sessions_df['discount_ratio'] > 0.66]['session_id'].tolist()))

        new_season_sessions_list = [str(session) for session in new_season_sessions_list]
        sales_sessions_list = [str(session) for session in sales_sessions_list]

        new_season_session_df = DataFrame(new_season_sessions_list, columns=['session_id'])
        new_season_session_df['session_type'] = '1'

        sales_sessions_session_df = DataFrame(sales_sessions_list, columns=['session_id'])
        sales_sessions_session_df['session_type'] = '0'

        session_type_df = concat([new_season_session_df, sales_sessions_session_df], ignore_index=True)
        session_type_df = session_type_df.merge(
            session_df[['session_id']].drop_duplicates().astype({'session_id': str}), on='session_id', how='outer')
        session_type_df.fillna('2', inplace=True)

        return session_type_df


class ProductsMetricsComputer:

    @staticmethod
    def compute_product_launch_date(df: DataFrame) -> DataFrame:
        """
        Given a DataFrame 'df' with columns [partnumber, timestamp_local, ...],
        computes, for each 'partnumber', the earliest (minimum) timestamp_local.

        Returns a DataFrame with columns [partnumber, first_appearance].
        """

        df = df.copy()
        if not api.types.is_datetime64_any_dtype(df['timestamp_local']):
            df['timestamp_local'] = to_datetime(df['timestamp_local'], errors='coerce')

        product_launch_date_df = df. \
            groupby(['partnumber', 'country'])['timestamp_local'].min().reset_index(name='product_launch_date')

        product_launch_date_df['product_launch_date'] = product_launch_date_df['product_launch_date'].dt.date

        return product_launch_date_df

    @staticmethod
    def add_freshness_column(products_df: DataFrame, reference_date: Timestamp = None) -> DataFrame:
        """
        Adds a 'freshness_bucket' column to products_df,
        which categorizes how "new" the product is relative to reference_date.

        E.g.:
          if product_launch_date within 7 days => 'very_new'
          else if within 30 days => 'new'
          else => 'older'
        """

        df = products_df.copy()

        if not api.types.is_datetime64_any_dtype(df['product_launch_date']):
            df['product_launch_date'] = to_datetime(df['product_launch_date'], errors='coerce')

        if reference_date is None:
            reference_date = to_datetime('2024-06-16')

        df['days_since_launch'] = (reference_date - df['product_launch_date']).dt.days.fillna(9999)

        def bucketize(days):
            if days <= 14:
                return 'new'
            else:
                return 'old'

        df['freshness_bucket'] = df['days_since_launch'].apply(bucketize)

        return df

    @staticmethod
    def compute_country_product_stats(train_df: DataFrame) -> DataFrame:
        """
        Returns a DataFrame with columns:
          - country
          - partnumber
          - visits_mean (average daily visits)
          - purchases_mean (average daily purchases)
          - ratio = 100 * purchases_mean / visits_mean (rounded to 2 decimals)

        This time, we ensure visits_mean & purchases_mean use the exact same number of days:
          1) Convert timestamp_local -> date
          2) Count daily visits & daily purchases by (country, partnumber, date)
          3) Outer merge daily visits & purchases => fill missing with 0
          4) Group by (country, partnumber): sum daily_visits & daily_purchases,
             count how many distinct days
          5) visits_mean = total_visits / days_count
             purchases_mean = total_purchases / days_count
        """

        df = train_df.copy()

        count_per_product_df = df.groupby(['partnumber']).size().reset_index(name='count')

        if not api.types.is_datetime64_any_dtype(df['timestamp_local']):
            df['timestamp_local'] = to_datetime(df['timestamp_local'], errors='coerce')
        df['date'] = df['timestamp_local'].dt.date

        daily_visits_df = df.groupby(['country', 'partnumber', 'date']).size().reset_index(name='daily_visits')

        purchases_df = df[df['add_to_cart'] > 0]
        daily_purchases_df = purchases_df. \
            groupby(['country', 'partnumber', 'date']) \
            .size() \
            .reset_index(name='daily_purchases')

        merged_daily_df =  \
            merge(daily_visits_df, daily_purchases_df, on=['country', 'partnumber', 'date'], how='outer'). \
            fillna(0)

        stats_agg = merged_daily_df. \
            groupby(['country', 'partnumber'], as_index=False). \
            agg(
                total_visits=('daily_visits', 'sum'),
                total_purchases=('daily_purchases', 'sum'),
                days_count=('date', 'count')
            )

        stats_agg['visits_mean'] = stats_agg['total_visits'] / stats_agg['days_count']
        stats_agg['purchases_mean'] = stats_agg['total_purchases'] / stats_agg['days_count']

        stats_agg = stats_agg.merge(count_per_product_df, on='partnumber', how='left')

        stats_agg['ratio'] = stats_agg. \
            apply(
                lambda row: round(100 * row['purchases_mean'] / row['visits_mean'], 2)
                if row['visits_mean'] > 0 else 0.0,
                axis=1
            )

        return stats_agg[['country', 'partnumber', 'visits_mean', 'purchases_mean', 'ratio', 'count']]

    @staticmethod
    def rank_products_by_ratio(df: DataFrame) -> DataFrame:
        """
        Adds a 'ratio_rank' column indicating how each product ranks by 'ratio'
        within its (cod_section, family) group, where rank=1 is the highest ratio.
        """
        df = df.sort_values(by=['cod_section', 'family', 'ratio'], ascending=[True, True, False]).copy()
        df['ratio_rank'] = df.groupby(['cod_section', 'family'])['ratio'].rank(method='dense', ascending=False)

        return df

    @staticmethod
    def apply_flag(group: DataFrame) -> DataFrame:
        v_quantile = group['visits_mean'].quantile(0.8)
        r_quantile = group['ratio'].quantile(0.8)
        group['popular'] = ((group['visits_mean'] >= v_quantile) & (group['ratio'] >= r_quantile)).astype(int)
        return group

    def flag_popular_products(self, df: DataFrame) -> DataFrame:
        """
        For each (cod_section, family) group, compute the 50th percentile of 'visits_mean'.
        Products with visits_mean >= that median get 'flag_popular'=1, else 0.
        Returns a new DataFrame with the additional column 'flag_popular'.
        """

        df = df.copy()
        grouped = df.groupby(['cod_section', 'family'], group_keys=False)
        df = grouped.apply(self.apply_flag).reset_index(drop=True)

        return df

    def main(self, train_df: DataFrame, products_df: DataFrame) -> DataFrame:

        train_df = train_df.astype(({'country': 'str'}))
        product_launch_date_df = self.compute_product_launch_date(train_df)

        product_metrics_list = []

        for country in ['25', '29', '34', '57']:
            country_df = train_df[train_df['country'] == country]

            country_stats_df = self.compute_country_product_stats(country_df)
            merged_df = country_stats_df.merge(product_launch_date_df, on=['partnumber', 'country'], how='left')
            merged_df['country'] = country

            product_metrics_list.append(merged_df)

        product_metrics_df = concat(product_metrics_list, ignore_index=True)
        product_metrics_df = product_metrics_df \
            .merge(products_df[['partnumber', 'cod_section', 'family', 'discount']], on='partnumber', how='left')

        product_metrics_df = self.rank_products_by_ratio(product_metrics_df)
        product_metrics_df = self.flag_popular_products(product_metrics_df)

        return product_metrics_df


def preprocess_data():
    """
    Main workflow for:
    1) Loading data
    2) Preprocess data
    3) Save data
    """

    train_df = read_csv('data/raw/train.csv')
    test_df = read_csv('data/raw/test.csv')
    user_df = read_csv('data/raw/user_data.csv')
    
    with open('data/raw/products.pkl', 'rb') as f:
        products_df = pickle.load(f)

    user_df = DataManager.expand_dataframe(user_df)
    train_df = Preprocessor().delete_long_sessions(train_df)
    train_df = Preprocessor().preprocess_data(products_df, train_df, user_df)
    corrected_train_df = Preprocessor().remove_mistake_clicks(train_df)
    test_df = Preprocessor().preprocess_data(products_df, test_df, user_df)
    eval_df, eval_session_product_map = Preprocessor().create_evaluation_subset(train_df, n_sessions=8000)

    test_discount_sessions_df, test_discount_length_df = ProductsInformationExtractor().compute_discount_ratio(test_df, products_df)
    session_type_df = ProductsInformationExtractor().compute_sessions_types(test_discount_sessions_df, test_discount_length_df, test_df)
    eval_discount_sessions_df, eval_discount_length_df = ProductsInformationExtractor().compute_discount_ratio(eval_df, products_df)
    eval_session_type_df = ProductsInformationExtractor().compute_sessions_types(eval_discount_sessions_df, eval_discount_length_df, eval_df)

    popular_products = ProductsInformationExtractor.get_popular_products(train_df, top_n=100)
    discount_map = ProductsInformationExtractor.create_discount_map(products_df)
    user_bought_map = ProductsInformationExtractor.build_user_bought_map(train_df)
    user_not_bought_map = ProductsInformationExtractor.build_user_not_bought_map(train_df)
    user_profiles = ProductsInformationExtractor.build_user_profiles(train_df)
    out_of_stock = ProductsInformationExtractor().compute_out_of_stock_products(train_df, 0, "2024-06-14 14:00:00")
    popular_family_products = ProductsInformationExtractor().build_section_family_products(train_df)
    product_metrics_df = ProductsMetricsComputer().main(corrected_train_df, products_df)

    user_df.to_csv('data/processed/user_data.csv', index=False)
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    eval_df.to_csv('data/processed/eval.csv', index=False)
    session_type_df.to_csv('data/processed/session_type.csv', index=False)
    eval_session_type_df.to_csv('data/processed/eval_session_type.csv', index=False)
    product_metrics_df.to_csv('data/processed/product_metrics_data.csv', index=False)
    with open('data/processed/popular_products.json', 'w') as archivo:
        json.dump(popular_products, archivo, indent=4)
    with open('data/processed/discount_map.json', 'w') as archivo:
        json.dump(discount_map, archivo, indent=4)
    with open('data/processed/eval_session_product_map.json', 'w') as archivo:
        json.dump(eval_session_product_map, archivo, indent=4)
    with open('data/processed/user_bought_map.json', 'w') as archivo:
        json.dump(user_bought_map, archivo, indent=4)
    with open('data/processed/user_not_bought_map.json', 'w') as archivo:
        json.dump(user_not_bought_map, archivo, indent=4)
    with open('data/processed/user_profiles.json', 'w') as archivo:
        json.dump(user_profiles, archivo, indent=4)
    with open('data/processed/out_of_stock.json', 'w') as archivo:
        json.dump(out_of_stock, archivo, indent=4)
    with open('data/processed/popular_family_products.json', 'w') as archivo:
        json.dump(popular_family_products, archivo, indent=4)


if __name__ == "__main__":
    preprocess_data()
