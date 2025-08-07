'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd

def load_data():
    model_pred_df = pd.read_csv('data/prediction_model_03.csv')
    model_pred_df.rename(columns={
        'actual genres': 'true_genres',
        'correct?': 'correct'
    }, inplace=True)
    genres_df = pd.read_csv('data/genres.csv')
    return model_pred_df, genres_df
    # Your code here


def process_data(model_pred_df, genres_df):
    genre_list = set()
    genre_true_counts = {}
    genre_tp_counts = {}
    genre_fp_counts = {}

    for index, row in model_pred_df.iterrows():
        true_genres = [g.strip().strip("[]'\"") for g in row['true_genres'].split(',')]
        pred_genres = [g.strip().strip("[]'\"") for g in row['predicted'].split(',')]


        genre_list.update(true_genres)
        genre_list.update(pred_genres)

        for genre in true_genres:
            genre_list.add(genre)
            genre_true_counts[genre] = genre_true_counts.get(genre, 0) + (1 if genre in true_genres else 0)
            genre_tp_counts[genre] = genre_tp_counts.get(genre, 0) + (1 if genre in pred_genres and genre in true_genres else 0)
            genre_fp_counts[genre] = genre_fp_counts.get(genre, 0) + (1 if genre not in true_genres and genre in pred_genres else 0)
    
    return list(genre_list), genre_true_counts, genre_tp_counts, genre_fp_counts

    # Your code here
