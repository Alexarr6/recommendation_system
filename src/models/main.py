import os
from itertools import product
from prepare_data import preprocess_data
from train_model import train
from predict_model import predict, evaluate


def main(
    no_components: int,
    percentile_visits: float,
    percentile_purchases: float,
    buy_threshold: float,
    time_threshold: float
) -> None:

    print('Preprocessing data...')
    preprocess_data()

    print('Computing similar products...')
    train(no_components, percentile_visits, percentile_purchases, buy_threshold)

    print('Evaluating model...')
    evaluate(no_components, percentile_visits, percentile_purchases, buy_threshold, time_threshold)

    print('Making recommendations...')
    predict(no_components, percentile_visits, percentile_purchases, buy_threshold, time_threshold)


if __name__ == "__main__":

    no_components_range = [220]
    percentile_visits_range = [0.5]
    percentile_purchases_range = [0.7]
    buy_threshold_range = [0.8]
    time_threshold_range = [2]

    param_combinations = list(product(no_components_range, percentile_visits_range, percentile_purchases_range, buy_threshold_range, time_threshold_range))

    for no_components, percentile_visits, percentile_purchases, buy_threshold, time_threshold in param_combinations:
        print(f"Testing with parameters: no_components={no_components}, percentile_visits={percentile_visits}, percentile_purchases={percentile_purchases}, buy_threshold={buy_threshold}, time_threshold={time_threshold}")
        main(no_components, percentile_visits, percentile_purchases, buy_threshold, time_threshold)
