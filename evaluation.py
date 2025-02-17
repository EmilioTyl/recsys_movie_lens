import pandas as pd 
import numpy as np

#Hit Rate => metric used in SLIM paper

def hit_rate(model, data_test, N=10, top_N =None):
    qty_users = len(data_test.user_id.unique())
    if top_N is None:
        top_N = model.all_useres_recomendation(N=N)
    
   
    hits_dict = {i: {"hits":0, "qty":0 }for i in range(1,6)}

    data_test = data_test.set_index("user_id")
    for u, l in zip(model.get_user_item().index, top_N):
        rating = data_test.loc[u]
        original_rate = rating["rating"]
        hits = hits_dict[original_rate]["hits"]
        qty = hits_dict[original_rate]["qty"]
        hits_dict[original_rate]["hits"] = hits+1 if rating["movie_id"] in l else hits
        hits_dict[original_rate]["qty"] = qty + 1
    data_test  = data_test.reset_index()
    hits = pd.DataFrame([ [original, h["hits"], h["qty"]] for original, h in hits_dict.items()], columns=["rate", "hits", "qty"])
    hits["hit_rate"] = hits["hits"] /hits["qty"]
    return hits

# Precision/Recall metric => metric used in "Performance of Recomender Algorithms"


def compute_precision_recall_by_N(model,top_N, data_test):
    N_max = top_N.shape[1]
    acum_recall = []
    acum_precision = []
    for n in range(1, N_max):
        computed_hits = compute_hits(model, top_N[:,:n], data_test)
        recall = computed_hits / data_test.shape[0]
        precision = recall/n
        acum_recall.append(recall)
        acum_precision.append(precision)
    return acum_recall, acum_precision

def compute_hits(model, top_N, data_test):
    hits = 0
    
    for i, r in enumerate(data_test.iterrows()):
        _, r = r
        movie_id = r["movie_id"]
        hits = hits+1 if movie_id in top_N[i,:] else hits
    return hits

# MAE / RMSE
def compute_rmse(y_pred, y_true):
    return np.sqrt(np.mean(np.power(y_pred - y_true, 2)))   
    
def compute_mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))
      
def evaluate_mae_rmse(estimate_f,data_train,data_test):
    """ RMSE-based predictive performance evaluation with pandas. """
    ids_to_estimate = zip(data_test.user_id, data_test.movie_id)
    estimated = np.array([estimate_f(u,i) if u in data_train.user_id else 3 for (u,i) in ids_to_estimate ])
    real = data_test.rating.values
    return compute_rmse(estimated, real), compute_mae(estimated, real)