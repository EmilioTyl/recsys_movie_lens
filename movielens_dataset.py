import pandas as pd
import numpy as np
import math

# Retrive raw dataset

def get_1M_movielens():
    # Load Data set
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('./data/ml-1m/users.dat', sep='::', names=u_cols)


    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('./data/ml-1m/ratings.dat', sep='::', names=r_cols)

    # the movies file contains columns indicating the movie's genres
    # let's only load the first three columns of the file with usecols
    m_cols = ['movie_id', 'title', 'category']
    movies = pd.read_csv('./data/ml-1m/movies.dat', sep='::', names=m_cols, usecols=range(3), encoding='latin-1')

    # Construcció del DataFrame
    data = pd.merge(pd.merge(ratings, users), movies)
    data = data[['user_id','title', 'movie_id','rating']]


    print("La BD has "+ str(data.shape[0]) +" ratings")
    print("La BD has ", data.user_id.nunique()," users")
    print("La BD has ", data.movie_id.nunique(), " movies")
    return data

def get_100K_movilens():
    # Load Data set
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv('./data/ml-100k/u.user', sep='|', names=u_cols)

    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('./data/ml-100k/u.data', sep='\t', names=r_cols)

    # the movies file contains columns indicating the movie's genres
    # let's only load the first three columns of the file with usecols
    m_cols = ['movie_id', 'title', 'release_date']
    movies = pd.read_csv('./data/ml-100k/u.item', sep='|', names=m_cols, usecols=range(3), encoding='latin-1')

    # Construcció del DataFrame
    data = pd.merge(pd.merge(ratings, users), movies)
    data = data[['user_id','title', 'movie_id','rating','release_date','sex','age']]


    print("La BD has "+ str(data.shape[0]) +" ratings")
    print("La BD has ", data.user_id.nunique()," users")
    print("La BD has ", data.movie_id.nunique(), " movies")
    return data

#Split and filter dataset

def assign_generator(test_size=0.1):
    def assign_to_set(df):
        sampled_ids = np.random.choice(df.index,
                                    size=np.int64(np.ceil(df.index.size * test_size)),
                                    replace=False)
        df.loc[sampled_ids, 'for_testing'] = True
        return df
    return assign_to_set


def assign_loocv(df):
    sampled_ids = np.random.choice(df.index)
    df.loc[[sampled_ids], 'for_testing'] = True
    return df

def remove_unrated(df, n=20):
    movie_stats = df.groupby('movie_id').agg({'rating': [np.size, np.mean]})
    min_n = movie_stats['rating']['size'] >= n
    filtered_movies_id = movie_stats[min_n].index
    print("Filtered movies ", len(filtered_movies_id.to_list()))
    df = df[df.movie_id.isin(filtered_movies_id)]
    return df

def split_train_val_by_movie(df, assign_=assign_loocv):
    print("Total movies", len(df.movie_id.unique()))
    movie_stats = df.groupby('movie_id').agg({'rating': [np.size, np.mean]})
    min_20 = movie_stats['rating']['size'] >= 20
    filtered_movies_id = movie_stats[min_20].index
    print("Filtered movies ", len(filtered_movies_id.to_list()))
    df = df[df.movie_id.isin(filtered_movies_id)]
    df.loc[:,'for_testing'] = False
    grouped = df.groupby('movie_id', group_keys=False).apply(assign_)
    data_train = df[grouped.for_testing == False]
    data_test = df[grouped.for_testing == True]

    print(data_train.shape )
    print(data_test.shape )

    print('Users:', df.user_id.nunique() )
    print('Movies:',df.movie_id.nunique() )
    return data_train, data_test

def split_train_val_by_user(df, assign_=assign_loocv):
    print("Total movies", len(df.movie_id.unique()))
    movie_stats = df.groupby('movie_id').agg({'rating': [np.size, np.mean]})
    min_20 = movie_stats['rating']['size'] >= 20
    filtered_movies_id = movie_stats[min_20].index
    print("Filtered movies ", len(filtered_movies_id.to_list()))
    df = df[df.movie_id.isin(filtered_movies_id)]

    df.loc[:,'for_testing'] = False
    grouped = df.groupby('user_id', group_keys=False).apply(assign_)
    data_train = df[grouped.for_testing == False]
    data_test = df[grouped.for_testing == True]

    print(data_train.shape )
    print(data_test.shape )

    print('Users:', df.user_id.nunique() )
    print('Movies:',df.movie_id.nunique() )
    return data_train, data_test

#Split used in paper "Performance of Recomender Algorithms"

def precision_recall_split(df, test_frac=0.014, rating_percent_thresh=0.33):
    df = remove_unrated(df, 20)
    data_test = df.sample(frac=test_frac)
    data_train = df.loc[~df.index.isin(data_test.index)]
    data_test = data_test[data_test.rating == 5]
    ordered_popularity = df.groupby("movie_id").agg({"rating": "count"}).sort_values(by="rating", ascending=False)
    total = ordered_popularity.rating.sum()
    ordered_popularity["rating_percent"] = ordered_popularity.rating / total
    ordered_popularity["rating_percent_acum"] = np.cumsum(ordered_popularity.rating_percent.to_numpy())
    
    filter_popular = ordered_popularity[ordered_popularity.rating_percent_acum < rating_percent_thresh]
    popular_movies = filter_popular.index
    T_head = data_test[data_test.movie_id.isin(popular_movies)]
    T_long = data_test[np.logical_not(data_test.movie_id.isin(popular_movies))]
    return data_train, data_test,T_head, T_long


#User-item matrix
def get_user_item(df):
    fileterd_df = df[["user_id", "movie_id", "rating"]]
    user_item = fileterd_df.pivot_table(index=['user_id'],columns=['movie_id'],values='rating')
    user_item.fillna(0, inplace = True )
    return user_item