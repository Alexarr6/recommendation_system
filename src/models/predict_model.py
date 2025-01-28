import os
import json
from typing import List, Dict

import pandas as pd
import numpy as np


def compute_avg_product_age_per_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns:
      - 'session_id': identifies each session
      - 'session_date': date/time of the session
      - 'product_launch_date': when the product was first made available
      - 'partnumber': product identifier

    1) Creates 'days_since_launch' = (session_date - product_launch_date) in days.
    2) Groups by session_id to find the average days_since_launch in that session.
    Returns a new DataFrame: [session_id, avg_days_since_launch].
    """

    # 1) Ensure both are datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    if not pd.api.types.is_datetime64_any_dtype(df['product_launch_date']):
        df['product_launch_date'] = pd.to_datetime(df['product_launch_date'], errors='coerce')

    df['days_since_launch'] = (df['date'] - df['product_launch_date']).dt.days

    avg_days_per_session = df.groupby('session_id')['days_since_launch'].mean().reset_index(
        name='avg_days_since_launch')
    count_per_session = df.groupby('session_id').size().reset_index(name='count')
    avg_days_per_session = avg_days_per_session.merge(count_per_session, on='session_id', how='inner')

    return avg_days_per_session


class MetricsService:
    """
    Holds methods for computing recommendation metrics such as NDCG.
    """
    @staticmethod
    def calculate_ndcg(
        session_recommendations: dict,
        eval_session_product_map: dict,
        k: int = 5
    ) -> float:
        """
        Calculates NDCG@k for a set of recommendations.
        session_recommendations: {session_id (as str): [recommended partnumbers]}
        user_session_df: The DataFrame containing actual interactions (add_to_cart > 0).
        """
        ndcg_scores = []

        for session_id in session_recommendations.keys():
            top_items = session_recommendations[session_id]
            relevant_items = eval_session_product_map[session_id]

            dcg = 0.0
            for idx, item in enumerate(top_items):
                if item in relevant_items:
                    dcg += 1 / np.log2(idx + 2)

            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_scores.append(ndcg)

        return np.mean(ndcg_scores)


class Recommender:
    """
    Contains the logic to generate recommendations based on time spent, discount info, and popularity.
    """
    def __init__(self, top_similar_dict: dict, top_similar_products_item_based: dict, bought_together_map: dict, popular_family_products: dict):
        self.top_similar_dict = top_similar_dict
        self.top_similar_products_item_based = top_similar_products_item_based
        self.bought_together_map = bought_together_map
        self.popular_family_products = popular_family_products

    @staticmethod
    def recommend_co_bought_items(
        viewed_products: List[int],
        recommendations: List[int],
        bought_together_map: Dict[int, List[int]],
        out_of_stock_country: List[int],
        N: int = 5
    ) -> List[int]:
        """
        Given a list of viewed_products in a session and a bought_together_map dict
        that shows which items are frequently purchased together,
        recommend up to N items from the mapping (union of all related items).
        """

        for product in viewed_products:
            if str(product) in bought_together_map:
                for related_item in bought_together_map[str(product)]:
                    if related_item not in recommendations and related_item not in out_of_stock_country:
                        recommendations.append(related_item)
                    if len(recommendations) >= N:
                        break
            if len(recommendations) >= N:
                break

        return recommendations

    @staticmethod
    def recommend_by_frequency(
        basket: List[int],
        recommendations: List[int],
        similar_dict: Dict[str, List[int]],
        out_of_stock_country: List[int],
        target_products: set,
        top_n: int = 5
    ) -> List[int]:
        """
        Recommends items that appear most frequently in the 'similar items' lists
        for every product in the basket.
        Excludes products already in the basket.
        """
        # Count how often each candidate item appears across the basket's similar lists
        frequency_map = {}

        for product_in_basket in basket:
            # Convert product_in_basket to string if your dict keys are str
            candidates = similar_dict.get(str(product_in_basket), [])
            candidates = [candidate for candidate in candidates if candidate not in out_of_stock_country]
            candidates = [candidate for candidate in candidates if candidate in target_products][:5]
            for candidate in candidates:
                if candidate not in recommendations and candidate:
                    frequency_map[candidate] = frequency_map.get(candidate, 0) + 1

        # Sort the candidate items by descending frequency
        # If two items have same frequency, you could tie-break by something else
        sorted_items = sorted(
            frequency_map.items(),
            key=lambda x: x[1],
            reverse=True
        )

        recommendations = recommendations + [item for item, freq in sorted_items[:top_n]]
        return recommendations

    @staticmethod
    def adding_user_profile_info(user_profile: dict) -> List[int]:

        not_bought_sorted = []
        category_fam_popularity = []
        for category_fam_key in user_profile.keys():
            famcod_data = user_profile.get(category_fam_key)
            if famcod_data['bought'] == 0:
                category_fam_popularity.append([category_fam_key, famcod_data['visits']])

        if category_fam_popularity:
            category_fam_popularity = sorted(category_fam_popularity, key=lambda x: x[1], reverse=True)
            not_bought_products = user_profile[category_fam_popularity[0][0]]['not_bought_products']
            not_bought_sorted = sorted(not_bought_products.items(), key=lambda x: x[1], reverse=True)
            not_bought_sorted = [int(product) for product, value in not_bought_sorted]

        return not_bought_sorted

    def recommend_products_for_session(
        self,
        session_data: pd.DataFrame,
        products_metrics_df: pd.DataFrame,
        top_products_for_country: dict,
        bought_together_map: dict,
        popular_products: List[int],
        discount_map: dict,
        user_profile: dict,
        out_of_stock_country: List[int],
        N: int = 5
    ) -> List[int]:
        """
        Generates up to N recommendations for a single session.
        1) Add top products (viewed) by count and mean time
        2) If not enough, add co bought products
        3) If not enough, add similar products
        4) If not enough, add similar products based on items
        5) If not enough, add most popular products of their family
        6) If still not enough, add popular products
        """

        session_type = session_data['session_type'].iloc[0]

        if session_type == 0:
            target_products = set([int(key) for key, value in discount_map.items() if value])
        elif session_type == 1:
            target_products = set([int(key) for key, value in discount_map.items() if not value])
        else:
            target_products = set([int(key) for key, value in discount_map.items()])

        session_data = session_data.merge(products_metrics_df[['partnumber', 'country', 'ratio']], on=['partnumber', 'country'], how='left')

        # Impute NaN times with the global mean
        global_mean_time = session_data['time_to_next_seconds'].mean(skipna=True)
        session_data = session_data.copy()
        session_data['time_to_next_seconds'] = session_data['time_to_next_seconds'].fillna(global_mean_time)

        # Build a DataFrame with 'count' and 'time_max'
        product_counts = session_data['partnumber'].value_counts()
        time_stats = session_data.groupby('partnumber')['time_to_next_seconds'].max()
        ratio_stats = session_data.groupby('partnumber')['ratio'].first()
        viewed_df = pd.DataFrame({'count': product_counts, 'time_max': time_stats, 'ratio': ratio_stats})
        viewed_df['discount'] = viewed_df.index.map(lambda p: discount_map.get(str(p), False))

        # Sort by count desc, then time_max desc
        viewed_sorted = viewed_df.sort_values(by=['count', 'ratio', 'time_max'], ascending=[False, False, True])
        viewed_products = list(viewed_sorted.index)

        recommendations = []

        # Step 1: Add top products (viewed) by count and mean time
        for product in viewed_products:
            recommendations.append(product)
            if len(recommendations) == N:
                return recommendations

        # Step 2: If not enough, add co bought products
        if len(recommendations) < N:
            recommendations = self.recommend_co_bought_items(
                viewed_products, recommendations, bought_together_map, out_of_stock_country,
                max(0, 5 - len(recommendations))
            )

        # Step 3: If not enough, add similar products
        if len(recommendations) < N:
            # Country recommendations
            recommendations = self.recommend_by_frequency(
                viewed_products, recommendations, top_products_for_country, out_of_stock_country, target_products,
                max(0, 5 - len(recommendations))
            )

        if len(recommendations) < N:
            viewed_not_bought_products = self.adding_user_profile_info(user_profile)
            for viewed_product in viewed_not_bought_products:
                if viewed_product not in recommendations and viewed_product not in out_of_stock_country and viewed_product in target_products:
                    recommendations.append(viewed_product)
                if len(recommendations) >= N:
                    break

        # Step 4: If not enough, add similar products based on items
        if len(recommendations) < N:
            recommendations = self.recommend_by_frequency(
                viewed_products, recommendations, self.top_similar_products_item_based, out_of_stock_country, target_products,
                max(0, 5 - len(recommendations))
            )

        # Step 5: If not enough, add most popular products of their family
        if len(recommendations) < N:
            for viewed_product in viewed_products:
                family = str(session_data[session_data['partnumber'] == viewed_product]['family'].values[0])
                if pd.isna(session_data[session_data['partnumber'] == viewed_product]['cod_section'].values[0]):
                    continue
                section = str(int(session_data[session_data['partnumber'] == viewed_product]['cod_section'].values[0]))
                key = section + '_' + family
                top_bought = self.popular_family_products[key]["top_viewed"]
                top_bought = [product for product, n in top_bought]
                for product in top_bought:
                    if product not in recommendations and product in target_products:
                        recommendations.append(product)
                    if len(recommendations) >= N:
                        break

        # Step 6: If still not enough, add popular products
        if len(recommendations) < N:
            for product in popular_products:
                if product not in recommendations and product not in out_of_stock_country and product in target_products:
                    recommendations.append(product)
                if len(recommendations) >= N:
                    break

        return recommendations

    def generate_session_recommendations(
        self,
        test_df: pd.DataFrame,
        products_metrics_df: pd.DataFrame,
        popular_products: List[int],
        discount_map: dict,
        user_profiles: dict,
        out_of_stock: dict,
        N: int = 5
    ) -> Dict[str, List[int]]:
        """
        Generates recommendations for each session in test_df, using:
        - popular_products
        - discount_map to incorporate discount info in ordering (if needed).
        """
        session_recommendations = {}

        for session_id in test_df['session_id'].unique():
            session_data = test_df[test_df['session_id'] == session_id]
            session_country = session_data['country'].iloc[0]
            top_products_for_country = self.top_similar_dict[str(session_country)]
            bought_together_map_for_country = self.bought_together_map
            out_of_stock_country = out_of_stock[str(session_country)]
            user_id = session_data['user_id'].iloc[0]
            user_profile = user_profiles.get(user_id, {})

            recs = self.recommend_products_for_session(
                session_data=session_data,
                products_metrics_df=products_metrics_df,
                top_products_for_country=top_products_for_country,
                bought_together_map=bought_together_map_for_country,
                popular_products=popular_products,
                discount_map=discount_map,
                out_of_stock_country=out_of_stock_country,
                user_profile=user_profile,
                N=N
            )

            session_recommendations[str(session_id)] = recs

        return session_recommendations


def predict(no_components: int, percentile_visits: float, percentile_purchases: float, buy_threshold: float, time_threshold: float) -> None:
    """
    Main workflow:
      1) Load data
      2) Filter by time_to_next_seconds >= 2
      3) Generate recommendations
      4) Save recommendations as JSON
    """
    # 1. Load data
    test_df = pd.read_csv('data/processed/test.csv')
    session_type_df = pd.read_csv('data/processed/session_type.csv')
    products_metrics_df = pd.read_csv('data/processed/product_metrics_data.csv')

    test_df = test_df. \
        merge(
            products_metrics_df[['partnumber', 'country', 'product_launch_date']],
            on=['partnumber', 'country'],
            how='left'
        )

    avg_days_per_session = compute_avg_product_age_per_session(test_df)
    test_df = test_df.merge(avg_days_per_session, on='session_id', how='inner')
    test_df = test_df.merge(session_type_df, on='session_id', how='inner')

    with open(f'data/processed/similar_products_lightfm/top_similar_products_{no_components}_{percentile_visits}_{percentile_purchases}.json', 'rb') as f:
        top_similar_products_lightfm = json.load(f)
    with open(f'data/processed/similar_products_item_based/top_similar_products_{percentile_visits}_{percentile_purchases}.json', 'rb') as f:
        top_similar_products_item_based = json.load(f)
    with open(f'data/processed/bought_together/bought_together_map_{buy_threshold}.json', 'rb') as f:
        bought_together_map = json.load(f)
    with open('data/processed/popular_products.json', 'rb') as f:
        popular_products = json.load(f)
    with open('data/processed/discount_map.json', 'rb') as f:
        discount_map = json.load(f)
    with open('data/processed/out_of_stock.json', 'rb') as f:
        out_of_stock = json.load(f)
    with open('data/processed/user_profiles.json', 'rb') as f:
        user_profiles = json.load(f)
    with open('data/processed/popular_family_products.json', 'rb') as f:
        popular_family_products = json.load(f)

    # 2. Filter by time >= 2 seconds
    global_mean_time = test_df['time_to_next_seconds'].mean(skipna=True)
    test_df['time_to_next_seconds'] = test_df['time_to_next_seconds'].fillna(global_mean_time)
    test_df = test_df[test_df['time_to_next_seconds'] >= time_threshold]

    # 3. Get popular products & generate recommendations
    session_recs = Recommender(
        top_similar_products_lightfm, top_similar_products_item_based, bought_together_map, popular_family_products
    ).generate_session_recommendations(
        test_df=test_df,
        products_metrics_df=products_metrics_df,
        popular_products=popular_products,
        discount_map=discount_map,
        user_profiles=user_profiles,
        out_of_stock=out_of_stock,
        N=5
    )

    # 4. Save recommendations
    recommendations_map = {'target': session_recs}
    with open(f'predictions/predictions_3.json', 'w') as archivo:
        json.dump(recommendations_map, archivo, indent=4)


def evaluate(no_components: int, percentile_visits: float, percentile_purchases: float, buy_threshold: float, time_threshold: float) -> None:
    """
    Main workflow:
      1) Load data
      2) Filter by time_to_next_seconds >= 2
      3) Generate recommendations
      4) Save evaluation as JSON
    """
    # 1. Load data
    eval_df = pd.read_csv('data/processed/eval.csv')
    session_type_df = pd.read_csv('data/processed/eval_session_type.csv')
    products_metrics_df = pd.read_csv('data/processed/product_metrics_data.csv')

    eval_df = eval_df. \
        merge(
            products_metrics_df[['partnumber', 'country', 'product_launch_date']],
            on=['partnumber', 'country'],
            how='left'
        )

    avg_days_per_session = compute_avg_product_age_per_session(eval_df)
    eval_df = eval_df.merge(avg_days_per_session, on='session_id', how='inner')
    eval_df = eval_df.merge(session_type_df, on='session_id', how='inner')

    with open(f'data/processed/similar_products_lightfm/top_similar_products_{no_components}_{percentile_visits}_{percentile_purchases}.json', 'rb') as f:
        top_similar_products_lightfm = json.load(f)
    with open(f'data/processed/similar_products_item_based/top_similar_products_{percentile_visits}_{percentile_purchases}.json', 'rb') as f:
        top_similar_products_item_based = json.load(f)
    with open(f'data/processed/bought_together/bought_together_map_{buy_threshold}.json', 'rb') as f:
        bought_together_map = json.load(f)
    with open('data/processed/popular_products.json', 'rb') as f:
        popular_products = json.load(f)
    with open('data/processed/discount_map.json', 'rb') as f:
        discount_map = json.load(f)
    with open('data/processed/eval_session_product_map.json', 'rb') as f:
        eval_session_product_map = json.load(f)
    with open('data/processed/out_of_stock.json', 'rb') as f:
        out_of_stock = json.load(f)
    with open('data/processed/user_profiles.json', 'rb') as f:
        user_profiles = json.load(f)
    with open('data/processed/popular_family_products.json', 'rb') as f:
        popular_family_products = json.load(f)

    file_score_path = 'predictions/models_scores.json'
    if os.path.exists(file_score_path):
        with open(file_score_path, 'r') as file:
            file_score_map = json.load(file)
    else:
        file_score_map = {}

    # 2. Filter by time >= 2 seconds
    eval_global_mean_time = eval_df['time_to_next_seconds'].mean(skipna=True)
    eval_df['time_to_next_seconds'] = eval_df['time_to_next_seconds'].fillna(eval_global_mean_time)
    eval_df = eval_df[eval_df['time_to_next_seconds'] >= time_threshold]

    # 3) Generate recommendations
    eval_session_recs = Recommender(
        top_similar_products_lightfm, top_similar_products_item_based, bought_together_map, popular_family_products
    ).generate_session_recommendations(
        test_df=eval_df,
        products_metrics_df=products_metrics_df,
        popular_products=popular_products,
        discount_map=discount_map,
        user_profiles=user_profiles,
        out_of_stock=out_of_stock,
        N=5
    )

    ndcg_value = MetricsService.calculate_ndcg(eval_session_recs, eval_session_product_map, k=5)
    print(
        f"NDCG@5 for no_components={no_components}, percentile_visits={percentile_visits}, percentile_purchases={percentile_purchases}, "
        f"buy_threshold={buy_threshold}, time_threshold={time_threshold}: {ndcg_value:.4f}"
    )

    # 4. Save evaluation
    prediction_file_name = f'predictions_3_{no_components}_{percentile_visits}_{percentile_purchases}_{buy_threshold}_{time_threshold}'
    file_score_map[prediction_file_name] = ndcg_value

    with open(file_score_path, 'w') as archivo:
        json.dump(file_score_map, archivo, indent=4)


if __name__ == "__main__":
    no_components = 50
    percentile_visits = 0.5
    percentile_purchases = 0.7
    buy_threshold = 0.8
    time_threshold = 2
    predict(no_components, percentile_visits, percentile_purchases, buy_threshold, time_threshold)
    evaluate(no_components, percentile_visits, percentile_purchases, buy_threshold, time_threshold)
