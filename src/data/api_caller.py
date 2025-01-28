import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import time


def make_api_request_call(user_id: str, url: str) -> dict:
    url = f"{url}/{user_id}"
    try:
        response = requests.get(url, headers={'Accept': 'application/json'}, timeout=10)
        if response.status_code == 200:
            return {'user_id': user_id, **response.json()['values']}
        else:
            return {'user_id': user_id, 'error': f'Failed to retrieve data. Status code: {response.status_code}'}
    except requests.exceptions.RequestException as e:
        return {'user_id': user_id, 'error': str(e)}


def add_error_users() -> None:
    error_df = pd.read_csv('data/raw/user_errors.csv')
    user_df = pd.read_csv('data/raw/user_data.csv')

    error_users_ids = error_df.user_id.to_list()

    error_user_data_list = []
    for user_id in error_users_ids:
        user_data = make_api_request_call(user_id, users_url)
        user_data['user_id'] = user_id
        error_user_data_list.append(user_data)

    error_user_df = pd.DataFrame(error_user_data_list)
    user_df = pd.concat([user_df, error_user_df])

    user_df.to_csv('data/raw/user_data.csv', index=False)


def fetch_all_user_data(user_ids: List[str], users_url: str, max_workers: int = 50) -> None:
    user_data_list = []
    errors = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_user_id = {
            executor.submit(make_api_request_call, user_id, users_url): user_id for user_id in user_ids
        }

        index = 0
        for future in as_completed(future_to_user_id):
            user_id = future_to_user_id[future]
            index += 1
            try:
                data = future.result()
                if 'error' in data:
                    errors.append(data)
                else:
                    user_data_list.append(data)
            except Exception as exc:
                errors.append({'user_id': user_id, 'error': str(exc)})

            if index % 5000 == 0:
                print(f'Processed {index} users')
                user_df = pd.DataFrame(user_data_list)
                user_df.to_csv('data/raw/user_data.csv', index=False)
                errors_df = pd.DataFrame(errors)
                errors_df.to_csv('data/raw/user_errors.csv', index=False)

    user_df = pd.DataFrame(user_data_list)
    user_df.to_csv('data/raw/user_data.csv', index=False)
    errors_df = pd.DataFrame(errors)
    errors_df.to_csv('data/raw/user_errors.csv', index=False)

    print(f'Total users processed: {index}')
    print(f'Total successful: {len(user_data_list)}')
    print(f'Total errors: {len(errors)}')


if __name__ == "__main__":
    users_url = 'https://zara-boost-hackathon.nuwe.io/users'

    response = requests.get(users_url, headers={'Accept': 'application/json'})
    users_ids = set(response.json())

    print('Total users:', len(users_ids))

    start_time = time.time()
    fetch_all_user_data(users_ids, users_url)
    end_time = time.time()

    print(f'Total time taken: {end_time - start_time} seconds')

    add_error_users()
