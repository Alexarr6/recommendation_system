import os
import json
import pickle
from typing import List, Dict

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import coo_matrix, csr_matrix
from pandas import DataFrame
from datetime import datetime


class DataManager:
    """
    Responsible for loading data and expanding rows.
    """

    @staticmethod
    def load_data(train_path: str, required_cols: List[str] = None) -> DataFrame:
        """
        Loads a DataFrame from the given CSV path, using the specified columns if provided.
        """
        if required_cols is None:
            required_cols = ['partnumber', 'date', 'timestamp_local', 'country', 'session_id', 'add_to_cart', 'cod_section', 'family']
        return pd.read_csv(train_path, usecols=required_cols)


class FeatureBuilder:
    """
    Utility class to build sets of features (user/item) and their corresponding matrices.
    """

    @staticmethod
    def create_item_features_set(products_df: DataFrame, item_feature_cols: List[str]) -> set:
        """
        Creates a set of all possible item features from the specified columns.
        e.g.: "discount:0", "discount:1", "color_id:83", ...
        """
        item_features = set()

        for col in item_feature_cols:
            unique_vals = products_df[col].dropna().unique()
            for val in unique_vals:
                item_features.add(f"{col}:{val}")

        return item_features

    @staticmethod
    def build_item_features_matrix(
        dataset: Dataset,
        product_ids: List[int],
        products_df: DataFrame,
        item_feature_cols: List[str]
    ):
        """
        Builds item_features_matrix for LightFM, matching the product_ids order.
        """
        product_df_indexed = products_df.set_index('partnumber').reindex(product_ids).reset_index().fillna('Unknown')

        features_list = []
        for _, row in product_df_indexed.iterrows():
            feature_values = []
            for col in item_feature_cols:
                val = row[col]
                feature_values.append(f"{col}:{val}")
            features_list.append((row['partnumber'], feature_values))

        item_features_matrix = dataset.build_item_features(features_list)
        return item_features_matrix

    @staticmethod
    def create_user_features_set(sessions_df: DataFrame, user_feature_cols: List[str]) -> set:
        """
        Creates a set of all possible user (session) features from the specified columns,
        e.g.: "device_type:2", "pagetype:home", ...
        """
        user_features = set()

        for col in user_feature_cols:
            unique_vals = sessions_df[col].dropna().unique()
            for val in unique_vals:
                user_features.add(f"{col}:{val}")

        return user_features

    @staticmethod
    def build_user_features_matrix(
        dataset: Dataset,
        user_ids: List[str],
        sessions_df: DataFrame,
        user_feature_cols: List[str]
    ):
        """
        Builds user_features_matrix for LightFM, matching the user_ids (session_ids) order.
        """
        user_df_indexed = DataFrame({'session_id': user_ids})
        user_df_indexed = user_df_indexed.merge(
            sessions_df[['session_id'] + user_feature_cols].drop_duplicates('session_id'),
            on='session_id',
            how='left'
        ).fillna('Unknown')

        features_list = []
        for _, row in user_df_indexed.iterrows():
            feature_values = []
            for col in user_feature_cols:
                val = row[col]
                feature_values.append(f"{col}:{val}")
            features_list.append((row['session_id'], feature_values))

        user_features_matrix = dataset.build_user_features(features_list)
        return user_features_matrix


class LightFMRecommender:
    """
    Responsible for building and using a LightFM recommendation model,
    as well as precomputing product similarities.
    """

    @staticmethod
    def _compute_country_product_stats(train_df: DataFrame) -> DataFrame:
        """
        Returns a DataFrame with columns:
          - country
          - partnumber
          - visits_mean (average daily visits)
          - purchases_mean (average daily purchases)

        Steps:
          1) Convert 'timestamp_local' to date.
          2) Count daily visits & daily purchases by (country, partnumber, date).
          3) Compute mean across days for each (country, partnumber).
        """

        df = train_df.copy()

        # 1) Ensure 'timestamp_local' is datetime, then extract date
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp_local']):
            df['timestamp_local'] = pd.to_datetime(df['timestamp_local'], errors='coerce')

        df['date'] = df['timestamp_local'].dt.date

        # 2) Count daily visits => group by (country, partnumber, date)
        daily_visits = (
            df.groupby(['country', 'partnumber', 'date'])
            .size()
            .reset_index(name='daily_visits')
        )
        visits_mean = (
            daily_visits
            .groupby(['country', 'partnumber'], as_index=False)['daily_visits']
            .mean()
            .rename(columns={'daily_visits': 'visits_mean'})
        )

        # 3) Count daily purchases => sub select rows where add_to_cart>0
        df_purchases = df[df['add_to_cart'] > 0]
        daily_purchases = (
            df_purchases.groupby(['country', 'partnumber', 'date'])
            .size()
            .reset_index(name='daily_purchases')
        )
        purchases_mean = (
            daily_purchases
            .groupby(['country', 'partnumber'], as_index=False)['daily_purchases']
            .mean()
            .rename(columns={'daily_purchases': 'purchases_mean'})
        )

        # 4) Merge visits_mean & purchases_mean
        stats = pd.merge(visits_mean, purchases_mean, on=['country', 'partnumber'], how='left').fillna(0)
        stats['ratio'] = round(100 * stats['purchases_mean'] / stats['visits_mean'], 2)

        return stats

    @staticmethod
    def _build_weighted_interactions(country_train_df: DataFrame):
        """
        Creates a generator of (user, item, weight) tuples, where weight=3 if add_to_cart>0, else weight=1.
        This way, products that were actually purchased get higher weight in the model.
        """
        for _, row in country_train_df.iterrows():
            session_id = row['session_id']
            partnumber = row['partnumber']
            if row['add_to_cart'] > 0:
                weight = 3
            else:
                weight = 1
            yield session_id, partnumber, weight

    @staticmethod
    def _train_or_load_lightfm(
        interactions: coo_matrix,
        model_path: str,
        item_features_matrix: csr_matrix = None,
        user_features_matrix: csr_matrix = None,
        no_components: int = 40,
        epochs: int = 50,
        num_threads: int = 10
    ):
        """
        Checks if the LightFM model at model_path exists.
        If so, loads it. Otherwise, trains a new model and saves it.
        """

        if os.path.exists(model_path):
            print(f"Loading LightFM model from {model_path}...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            print(f"No existing model found. Training {model_path} model...")
            model = LightFM(no_components=no_components, loss='warp', random_state=1)
            model.fit(
                interactions=interactions,
                item_features=item_features_matrix,
                user_features=user_features_matrix,
                epochs=epochs,
                num_threads=num_threads
            )
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

        return model

    def train_lightfm_model(
        self,
        train_df: DataFrame,
        products_df: DataFrame,
        country: int,
        user_feature_cols: List[str],
        item_feature_cols: List[str],
        no_components: int,
        percentile_visits: float = 0.6,
        percentile_purchases: float = 0.6
    ) -> Dict[int, List[int]]:
        """
        Trains a LightFM model for a given country and returns the product similarity dictionary.
        """
        country_train_df = train_df[train_df['country'] == country]

        item_features_set = FeatureBuilder.create_item_features_set(products_df, item_feature_cols)
        user_features_set = FeatureBuilder.create_user_features_set(country_train_df, user_feature_cols)

        dataset = Dataset()
        dataset.fit(
            users=country_train_df['session_id'].unique(),
            items=country_train_df['partnumber'].unique(),
            user_features=user_features_set,
            item_features=item_features_set
        )

        interactions, _ = dataset.build_interactions(
            self._build_weighted_interactions(country_train_df)
        )

        item_id_mapping, _ = dataset.mapping()[2:]
        index_to_product_id = {v: k for k, v in item_id_mapping.items()}
        product_ids = [index_to_product_id[i] for i in range(len(index_to_product_id))]

        user_id_mapping, _, _, _ = dataset.mapping()
        index_to_user_id = {v: k for k, v in user_id_mapping.items()}
        user_ids = [index_to_user_id[i] for i in range(len(index_to_user_id))]

        item_features_matrix = FeatureBuilder.build_item_features_matrix(
            dataset,
            product_ids,
            products_df,
            item_feature_cols
        )
        user_features_matrix = FeatureBuilder.build_user_features_matrix(
            dataset,
            user_ids,
            country_train_df,
            user_feature_cols
        )

        model = self._train_or_load_lightfm(
            interactions,
            f'models/lightfm_model_{country}_{no_components}.pkl',
            item_features_matrix,
            user_features_matrix,
            no_components
        )

        product_embeddings = model.item_embeddings

        similar_products = self._precompute_similar_products_with_embeddings(
            product_embeddings, product_ids, country_train_df, top_n=20, percentile_visits=percentile_visits, percentile_purchases=percentile_purchases
        )

        return similar_products

    def _precompute_similar_products_with_embeddings(
        self,
        product_embeddings: np.ndarray,
        product_ids: list,
        train_df: DataFrame,
        top_n: int = 5,
        percentile_visits: float = 0.6,
        percentile_purchases: float = 0.6
    ) -> Dict[int, List[int]]:
        """
        Computes top-N similar products using item embeddings, but uses
        *per-category* percentile thresholds for visits_mean and purchases_mean
        instead of a global min_visits/min_purchases.

        Steps:
        1) Compute stats with self._compute_country_product_stats(...) => columns [country, partnumber, visits_mean, purchases_mean].
        2) Group by (cod_section, family) or any category grouping you want,
           find the local percentile threshold for visits_mean & purchases_mean.
        3) Keep items above those local thresholds in each group.
           Union them => 'popular_products'.
        4) Use only those items to build the candidate pool for similarity.
        5) Return a dict {product_id: [top_sim]}.
        """

        # 1) get stats => includes columns visits_mean, purchases_mean
        country_stats = self._compute_country_product_stats(
            train_df[['partnumber', 'timestamp_local', 'country', 'add_to_cart']]
        )

        if 'cod_section' not in country_stats.columns or 'family' not in country_stats.columns:
            extra_info = train_df[['country', 'partnumber', 'cod_section', 'family']].drop_duplicates()
            country_stats = pd.merge(country_stats, extra_info, on=['country', 'partnumber'], how='left')

        # 3) group by (cod_section, family) => find local percentile
        def local_percentile(group, col, percentile):
            return group[col].quantile(percentile)

        popular_products_set = set()

        grouping_cols = ['country', 'cod_section', 'family']
        grouped = country_stats.groupby(grouping_cols)

        for (country, codsec, fam), group_df in grouped:
            if group_df.empty:
                continue
            # local thresholds
            v_threshold = local_percentile(group_df, 'visits_mean', percentile_visits)
            p_threshold = local_percentile(group_df, 'purchases_mean', percentile_purchases)
            r_threshold = local_percentile(group_df, 'ratio', 0.15)

            # keep rows that exceed these local thresholds
            qualified = group_df[
                (group_df['visits_mean'] >= v_threshold) &
                (group_df['purchases_mean'] >= p_threshold) &
                (group_df['ratio'] >= r_threshold)
                ]['partnumber'].unique()

            # add to global set
            for pid in qualified:
                popular_products_set.add(pid)

        # 4) now build your candidate set from those popular products
        filtered_indices = [i for i, pid in enumerate(product_ids) if pid in popular_products_set]

        if not filtered_indices:
            return {}

        candidate_embeddings = product_embeddings[filtered_indices]
        candidate_ids = [product_ids[i] for i in filtered_indices]

        # 5) compute normalized embeddings
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        candidate_normalized = candidate_embeddings / candidate_norms

        all_norms = np.linalg.norm(product_embeddings, axis=1, keepdims=True)
        all_normalized = product_embeddings / all_norms

        # 6) build top_similar_dict for each product
        top_similar_dict = {}

        for i, product_id in enumerate(product_ids):
            base_vec = all_normalized[i:i + 1]
            similarities = np.dot(base_vec, candidate_normalized.T)[0]
            sim_indices = np.argsort(-similarities)[:top_n]
            top_candidates = [int(candidate_ids[idx]) for idx in sim_indices]
            top_similar_dict[int(product_id)] = top_candidates

        return top_similar_dict

    def run(
        self,
        train_df: DataFrame,
        products_df: DataFrame,
        no_components: int,
        percentile_visits: float,
        percentile_purchases: float,
        item_feature: List[str],
        user_feature: List[str],
        save_path: str
    ) -> None:
        """
        Main workflow for:
        1) Loading data
        2) Filtering sessions
        3) Computing time between products
        4) Training LightFM model for specific countries
        5) Saving the output
        """
        model_train_df = train_df.copy()
        model_products_df = products_df.copy()

        similar_products_25 = self.train_lightfm_model(
            model_train_df, model_products_df, 25, user_feature, item_feature, no_components, percentile_visits, percentile_purchases
        )
        similar_products_29 = self.train_lightfm_model(
            model_train_df, model_products_df, 29, user_feature, item_feature, no_components, percentile_visits, percentile_purchases
        )
        similar_products_34 = self.train_lightfm_model(
            model_train_df, model_products_df, 34, user_feature, item_feature, no_components, percentile_visits, percentile_purchases
        )
        similar_products_57 = self.train_lightfm_model(
            model_train_df, model_products_df, 57, user_feature, item_feature, no_components, percentile_visits, percentile_purchases
        )

        top_similar_products_by_country = {
            '25': similar_products_25,
            '29': similar_products_29,
            '34': similar_products_34,
            '57': similar_products_57
        }

        with open(save_path, 'w') as f:
            json.dump(top_similar_products_by_country, f, indent=4)


class BoughtTogetherComputer:

    @staticmethod
    def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """
        Returns a matrix of cosine similarities between all item embeddings.
        embeddings shape: (num_products, embedding_dim)
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms

        similarity_matrix = np.dot(normalized, normalized.T)
        return similarity_matrix

    @staticmethod
    def filter_low_usage_items(df: DataFrame, min_avg_visits_per_day: float = 1.0) -> DataFrame:
        """
        1) Compute average daily usage for each 'partnumber'.
        2) Keep only items with >= min_avg_visits_per_day.
        Returns the filtered df.
        Assumes df has columns ['partnumber', 'timestamp_local'] at least.
        """
        temp = df.copy()

        # Ensure timestamp_local is datetime
        if not pd.api.types.is_datetime64_any_dtype(temp['timestamp_local']):
            temp['timestamp_local'] = pd.to_datetime(temp['timestamp_local'], errors='coerce')

        # Extract date
        temp['date'] = temp['timestamp_local'].dt.date

        # Count daily usage (all rows) => group by partnumber, date
        daily_usage = temp.groupby(['partnumber', 'date']).size().reset_index(name='daily_count')

        # Then compute mean daily_count
        avg_usage = daily_usage.groupby('partnumber')['daily_count'].mean().reset_index(name='avg_daily_count')

        # Filter to keep only items with >= threshold
        high_usage_items = avg_usage[avg_usage['avg_daily_count'] >= min_avg_visits_per_day]['partnumber'].unique()
        filtered_df = temp[temp['partnumber'].isin(high_usage_items)].copy()

        return filtered_df

    def train_bought_together_model(self, train_df: DataFrame, min_avg_visits_per_day: float) -> tuple:

        high_usage_df = self.filter_low_usage_items(df=train_df, min_avg_visits_per_day=min_avg_visits_per_day)
        train_buys_df = high_usage_df[high_usage_df['add_to_cart'] > 0].copy()

        dataset = Dataset()
        dataset.fit(users=train_buys_df['session_id'].unique(), items=train_buys_df['partnumber'].unique())

        interactions, _ = dataset.build_interactions(
            (row['session_id'], row['partnumber']) for _, row in train_buys_df.iterrows()
        )

        item_id_mapping, _ = dataset.mapping()[2:]
        index_to_product_id = {v: k for k, v in item_id_mapping.items()}
        product_ids = [index_to_product_id[i] for i in range(len(index_to_product_id))]

        model = LightFM(no_components=40, loss='warp', random_state=1)
        model.fit(interactions, epochs=50, num_threads=10)

        product_embeddings = model.item_embeddings

        return product_ids, product_embeddings

    @staticmethod
    def map_indices_above_threshold(
        similarity_matrix: np.ndarray,
        index_to_pid: Dict[int, int],
        threshold: float
    ) -> Dict[int, List[int]]:
        """
        Returns a dict {product_id: [list_of_related_product_ids]}
        where similarity > threshold.
        """
        num_products = similarity_matrix.shape[0]
        bought_together_map = {}

        for i in range(num_products):
            pid_i = index_to_pid[i]
            related = []
            for j in range(num_products):
                if j == i:
                    continue
                if similarity_matrix[i, j] > threshold:
                    pid_j = index_to_pid[j]
                    related.append(int(pid_j))
            if related:
                bought_together_map[int(pid_i)] = related

        return bought_together_map

    def compute_bought_together_products(
        self,
        train_df: DataFrame,
        threshold: float,
        min_avg_visits_per_day: float
    ) -> Dict[int, List[int]]:

        product_ids, product_embeddings = self.train_bought_together_model(train_df, min_avg_visits_per_day)
        sim_matrix = self.compute_similarity_matrix(product_embeddings)
        index_to_product_id = {i: pid for i, pid in enumerate(product_ids)}
        bought_together_map = self.map_indices_above_threshold(sim_matrix, index_to_product_id, threshold)

        return bought_together_map


def train(no_components: int, percentile_visits: float, percentile_purchases: float, buy_threshold: float) -> None:
    """
    Main workflow for:
    1) Loading data
    2) Preprocess data
    3) Save data
    """

    train_df = DataManager.load_data('data/processed/train.csv')

    with open('data/raw/products.pkl', 'rb') as f:
        products_df = pickle.load(f)

    similar_products_item_based_path = f'data/processed/similar_products_item_based/top_similar_products_{percentile_visits}_{percentile_purchases}.json'
    similar_products_lightfm_path = f'data/processed/similar_products_lightfm/top_similar_products_{no_components}_{percentile_visits}_{percentile_purchases}.json'
    bought_together_path = f'data/processed/bought_together/bought_together_map_{buy_threshold}.json'

    if not os.path.exists(similar_products_item_based_path):
        filtered_products_df = products_df.dropna(subset=['embedding']).copy()
        embeddings = np.vstack(filtered_products_df['embedding'].to_numpy())
        product_ids = filtered_products_df['partnumber'].to_numpy()

        top_similar_item_based_dict = LightFMRecommender()._precompute_similar_products_with_embeddings(
            embeddings, product_ids, train_df, 150, percentile_visits, percentile_purchases
        )

        with open(similar_products_item_based_path, 'w') as f:
            json.dump(top_similar_item_based_dict, f, indent=4)

    if not os.path.exists(similar_products_lightfm_path):
        lightfm_recommender = LightFMRecommender()
        lightfm_recommender.run(
            train_df,
            products_df,
            no_components,
            percentile_visits,
            percentile_purchases,
            [],
            [],
            similar_products_lightfm_path
        )

    if not os.path.exists(bought_together_path):
        bought_together_map = BoughtTogetherComputer().compute_bought_together_products(train_df, buy_threshold, 20)

        with open(bought_together_path, 'w') as f:
            json.dump(bought_together_map, f, indent=4)


if __name__ == "__main__":
    no_components = 220
    percentile_visits = 0.5
    percentile_purchases = 0.7
    buy_threshold = 0.8
    train(no_components, percentile_visits, percentile_purchases, buy_threshold)
