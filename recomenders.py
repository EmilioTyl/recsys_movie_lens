import pandas as pd
import numpy as np
import math
from sklearn.metrics import pairwise_distances
import pdb
from scipy.spatial.distance import cosine
from sklearn.linear_model import SGDRegressor, ElasticNet
from sklearn.metrics import mean_squared_error
import scipy.sparse as sparse



## SLIM

class SLIM():
    def __init__(self, l1_reg=0.001, l2_reg=0.0001):
        alpha = l1_reg+l2_reg
        l1_ratio = l1_reg/alpha
        ignore_negative_weights=False,
        fit_intercept=False
        self.model = ElasticNet(alpha=alpha,l1_ratio=l1_ratio,positive=True,fit_intercept=fit_intercept,copy_X=False)#SGDRegressor(penalty='elasticnet',fit_intercept=fit_intercept,alpha=0.0001,l1_ratio=0.15)
    
    def fit(self, user_item):
        self.user_item = user_item
        self.sparse_user_item = sparse.csc_matrix(self.user_item)
        row = []
        col = []
        data = []
        loss = []
        user_qty, item_qty = self.sparse_user_item.shape
        for j in range(item_qty):
            if j%500 == 0 and j!=0:
                print("Training for item ",j)
                print("loss ", np.array(loss).mean())
            a_j = self.sparse_user_item.getcol(j).copy()
            f, t = self.sparse_user_item.indptr[j:j+2]
            original = self.sparse_user_item.data[f:t].copy()
            self.sparse_user_item.data[f:t] = np.zeros(len(original))

            self.model.fit(self.sparse_user_item, a_j.toarray().ravel())
            y_pred = self.model.predict(self.sparse_user_item)
            loss.append(mean_squared_error(y_pred, a_j.toarray().ravel()))

            self.sparse_user_item.data[f:t] = original

            weights = self.model.coef_

            for i, w in enumerate(weights):
                if w!=0 :
                    row.append(j)
                    col.append(i)
                    data.append(w)
        self.sparse_user_item = sparse.csr_matrix(self.sparse_user_item)
        self.W = sparse.csr_matrix((data,(row, col)),(item_qty, item_qty))
        self.training_loss = np.array(loss).mean()
        self.sparcity = self.W.nnz / (self.W.shape[0] * self.W.shape[1])
        

    
    def user_recomendation(self, u, N=10):
        self.movie_columns = self.user_item.columns
        users_index = np.argwhere(self.user_item.index.isin([u])).ravel()
        ratings = (self.sparse_user_item[users_index] * self.W.T ).toarray().flatten()
        rated_items = set(self.sparse_user_item[users_index].indices)
        top_N = [(i,ratings[i]) for i in ratings.argsort()[::-1] if i not in rated_items][:N]
        top_N_viewed = [(self.movie_columns[i],ratings[i],self.sparse_user_item[users_index, i].data) for i in ratings.argsort()[::-1]][:N]
        return top_N, top_N_viewed
    
    def all_useres_recomendation(self, N=10, val_rating=False):
        self.movie_columns = self.user_item.columns
        ratings = self.sparse_user_item * self.W.T 
        
        top_N = []
 
        max_score = ratings.data.max() * np.ones(self.sparse_user_item.nnz)
        elim_matrix = self.sparse_user_item.copy()
        elim_matrix.data = max_score
        
        ratings = ratings - elim_matrix # set to rating<0 to the movies that are already rated
        for u in range(self.sparse_user_item.shape[0]):
            r = ratings[u,:]
            if val_rating:
                top_N.append([(self.movie_columns[i],v) for v,i in sorted(zip(r.data,r.indices),reverse=True) if v > 0][:N])
            else:
                top_N.append([ self.movie_columns[i] for v,i in sorted(zip(r.data,r.indices),reverse=True) if v > 0][:N])
        return np.array(top_N)
    
    def all_useres_recomendation_precision_recall(self, selection=1000, N=10):
        self.movie_columns = self.user_item.columns
        ratings = self.sparse_user_item * self.W.T 
        
        top_N = []
 
        max_score = ratings.data.max() * np.ones(self.sparse_user_item.nnz)
        elim_matrix = self.sparse_user_item.copy()
        elim_matrix.data = max_score
        
        ratings = ratings - elim_matrix # set to rating<0 to the movies that are already rated
        for u in range(self.sparse_user_item.shape[0]):
            r = ratings[u,:]
            idx = np.random.permutation(len(r.data))[:selection]
            selected_list = np.array([ self.movie_columns[i] for v,i in sorted(zip(r.data[idx],r.indices[idx]),reverse=True) if v > 0])
            selected_list = selected_list[:N]
            top_N.append(selected_list)
        return np.array(top_N)

    def estimate(self, user_id, movie_id):
        u, = np.where(self.user_item.index==user_id)
        m, = np.where(self.user_item.columns==movie_id)
        ratings = self.sparse_user_item[u[0],:] * self.W.T 
        rating = ratings[:,m[0]]
        return rating.data[0] if len(rating.data) > 0 else 0

    def get_user_item(self):
        return self.user_item

## KNN recomender


def custom_cosine(x,y):
    mask = (x !=0) & (y!=0)
    if any(mask):
        d = 1-cosine(x[mask], y[mask])
        return d if d>0 else 0
    else:
        return 0
    
class ItemBasedCF:
    def __init__(self, knei=None):
        self.knei = knei
        
    def fit(self, data_train, normalize=True):
        self.df = data_train
        
        if normalize:
            user_mean = pd.DataFrame(data_train.groupby('user_id')["rating"].mean()).reset_index()
            user_mean = user_mean.rename(columns={"rating": "mean_rating"})
            data_train = pd.merge(data_train, user_mean, on="user_id")
            data_train["normalized_rating"] = data_train["rating"] - data_train["mean_rating"] #Normalize
            fileterd_df = data_train[["user_id", "movie_id", "rating", "normalized_rating"]]
            self.user_item = fileterd_df.pivot_table(index=['movie_id'],columns=['user_id'],values='normalized_rating')#.reset_index(drop=True)
        else:
            fileterd_df = data_train[["user_id", "movie_id", "rating"]]
            self.user_item = fileterd_df.pivot_table(index=['movie_id'],columns=['user_id'],values='rating')#.reset_index(drop=True)
        self.user_item.fillna(0, inplace = True )
        index = self.user_item.index
        
        movie_similarity = pairwise_distances( self.user_item.to_numpy(), metric=custom_cosine)
        np.fill_diagonal(movie_similarity, 0 ) #Filling diagonals with 0s for future use when sorting is done
        self.sim_matrix = pd.DataFrame(movie_similarity)
        self.sim_matrix.index = index
        self.sim_matrix.columns = index.to_list()
        if normalize: self.user_item = fileterd_df.pivot_table(index=['movie_id'],columns=['user_id'],values='rating')
        self.sparse_user_item = sparse.csr_matrix(self.user_item.fillna(0).T)
        self.sim_matrix_np = movie_similarity
        
    def estimate_df(self, df):
        e = df[["user_id", "movie_id"]]
        estimated_rate = []
        for i,r in e.iterrows():
            rate = self.estimate(r["user_id"], r["movie_id"])
            estimated_rate.append(rate)
        return np.array(estimated_rate)
    
    def estimate(self, user_id, movie_id):
        filtered = self.user_item[self.user_item[user_id]>0][user_id]
        rated_movies = (filtered).index.to_list()

        if movie_id in self.sim_matrix.index:
            weights = self.sim_matrix.loc[movie_id, rated_movies].to_numpy()
            arg_sort = np.argsort(weights)[::-1]
        else:
            return filtered.mean()
        num = np.multiply(filtered.to_numpy()[arg_sort],weights[arg_sort])
        if self.knei:
            num = num[:self.knei].sum()
            den = weights[arg_sort][:self.knei].sum()
        else:
            num = num.sum()
            den = weights.sum()
        rate = num/den
        return rate
    
    def user_recomendation(self, user_index, selection = None):
        user_sparse_vector = self.sparse_user_item[user_index]
        rated_items = self.sparse_user_item[user_index].indices


        movies_to_rate_mask = np.ones(self.sim_matrix_np.shape[0], dtype=bool)
        movies_to_rate_mask[rated_items] = False
        if selection:
            i,= np.where(movies_to_rate_mask)
            i = np.random.choice(i,selection,replace=False)
            movies_to_rate_mask[i]=False
        movies_to_rate = self.sim_matrix_np[:,movies_to_rate_mask]
        movies_to_rate = movies_to_rate[rated_items,:]

        if not self.knei:
            num = (user_sparse_vector.data).dot(movies_to_rate)
            den = movies_to_rate.sum(axis=0)
            ratings = np.divide(num, den)
        else:
            qty = movies_to_rate.shape[1]
            ratings = np.zeros(qty)
            for j in range(qty):
                weights = movies_to_rate[:,j]
                arg_sort = np.argsort(weights)[::-1]
                user = user_sparse_vector.data[arg_sort]
                weights = weights[arg_sort]
                ratings[j] = user.dot(weights)/weights.sum()
        index_movies, = np.where(movies_to_rate_mask)
        return [self.sim_matrix.columns[m_ix] for r, m_ix in sorted(zip(ratings, index_movies), reverse=True)]
        
    def all_useres_recomendation(self, N=10, selection=None):
        user_indexes = self.user_item.index
        recomendation = []
        for u_index in range(self.sparse_user_item.shape[0]):
            recomendation.append(self.user_recomendation(u_index, selection=selection)[:N])
        return np.array(recomendation)

    def all_useres_recomendation_precision_recall(self, selection=300, N=10):
        return self.all_useres_recomendation(N=10, selection=selection)
    
    def get_user_item(self):
        return self.user_item.T
    

### PureSVD


class PureSVD():
    def __init__(self, latent_factors):
        self.latent_factors = latent_factors

    def fit(self, user_item):
        self.user_item = user_item
        self.sparse_user_item = sparse.csr_matrix(self.user_item)
   

        u, s, qt = sparse.linalg.svds(self.sparse_user_item, k=self.latent_factors)

        self.Q = qt.T
        self.QT = qt
        self.P = u * (sparse.identity(self.latent_factors).multiply(s))

        self.sparse_user_item = sparse.csr_matrix(self.sparse_user_item)
    
        

    
    def user_recomendation(self, u, N=10):
        self.movie_columns = self.user_item.columns
        users_index = np.argwhere(self.user_item.index.isin([u])).ravel()
        ratings = (self.sparse_user_item[users_index] * self.Q * self.QT ).toarray().flatten()
        rated_items = set(self.sparse_user_item[users_index].indices)
        top_N = [(i,ratings[i]) for i in ratings.argsort()[::-1] if i not in rated_items][:N]
        top_N_viewed = [(self.movie_columns[i],ratings[i],self.sparse_user_item[users_index, i].data) for i in ratings.argsort()[::-1]][:N]
        return top_N, top_N_viewed
    
    def all_useres_recomendation(self, N=10, val_rating=False):
        self.movie_columns = self.user_item.columns
        ratings = sparse.csr_matrix((self.sparse_user_item * self.Q).dot(self.QT))
        
        top_N = []
 
        max_score = ratings.data.max() * np.ones(self.sparse_user_item.nnz)
        elim_matrix = self.sparse_user_item.copy()
        elim_matrix.data = max_score
        
        ratings = ratings - elim_matrix # set to rating<0 to the movies that are already rated
        for u in range(self.sparse_user_item.shape[0]):
            r = ratings[u,:]
            if val_rating:
                top_N.append([(self.movie_columns[i],v) for v,i in sorted(zip(r.data,r.indices),reverse=True) if v > 0][:N])
            else:
                top_N.append([ self.movie_columns[i] for v,i in sorted(zip(r.data,r.indices),reverse=True) if v > 0][:N])
        return np.array(top_N)
    
    def all_useres_recomendation_precision_recall(self, selection=1000, N=10):
        self.movie_columns = self.user_item.columns
        ratings = sparse.csr_matrix((self.sparse_user_item * self.Q).dot(self.QT))
        
        top_N = []
 
        max_score = ratings.data.max() * np.ones(self.sparse_user_item.nnz)
        elim_matrix = self.sparse_user_item.copy()
        elim_matrix.data = max_score
        
        ratings = ratings - elim_matrix # set to rating<0 to the movies that are already rated
        for u in range(self.sparse_user_item.shape[0]):
            r = ratings[u,:]
            idx = np.random.permutation(len(r.data))[:selection]
            selected_list = np.array([ self.movie_columns[i] for v,i in sorted(zip(r.data[idx],r.indices[idx]),reverse=True) if v > 0])
            selected_list = selected_list[:N]
            top_N.append(selected_list)
        return np.array(top_N)

    def get_user_item(self):
        return self.user_item  

    def estimate(self, user_id, movie_id):
        u,=np.where(self.user_item.index==user_id)
        m,=np.where(self.user_item.columns==movie_id)
        rating = sparse.csr_matrix((self.sparse_user_item[u,:] * self.Q).dot(self.QT[:,m]))
        return rating.data[0] if len(rating.data) > 0 else 0


