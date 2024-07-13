import numpy as np
import pandas as pd
import clustbench
import pickle
from sklearn.metrics import adjusted_rand_score as ari
from genieclust.cluster_validity import generalised_dunn_index as gdi
from sklearn.model_selection import StratifiedKFold

data_path = "clustering-data-v1-1.1.0/"
res_path  = "benchmark/"

# Setting up the results dataframe
timing = pd.read_csv(res_path+"timing.txt", names=[
    "battery","dataset","model","time","time_rf","time_aff","time_clus"])
results = timing[["battery","dataset"]].drop_duplicates().reset_index(drop=True)

# Add timing information to results
for col in ["time","time_rf","time_aff","time_clus"]:
    for model in ["old","new"]:
        results[f"{model}_{col}"] = timing.loc[timing.model == model, col].values

for i,dataset in enumerate(results.dataset):
    # Load the dataset
    battery = results.battery[i]
    benchmark = clustbench.load_dataset(battery, dataset, data_path)
    X = benchmark.data
    y = benchmark.labels
    # Add dataset information to results
    results.loc[i,"n"] = X.shape[0]
    results.loc[i,"p"] = X.shape[1]
    results.loc[i,"kmin"] = min(benchmark.n_clusters)
    results.loc[i,"kmax"] = max(benchmark.n_clusters)
    # Subsample if necessary
    if (battery == "mnist") or ((battery == "sipu") and (dataset in 
                                ["birch1","birch2","worms_2","worms_64"])):
        kfold = StratifiedKFold(n_splits=10) # 10% of the data
        ilabels = np.argmax(benchmark.n_clusters) # take labeling with most number of clusters
        iloc = list(kfold.split(X, y[ilabels]))[0][1]
        X = X[iloc,:]
        y = [labeling[iloc] for labeling in y]
    # Compute aditional measures
    fname = lambda model: res_path+f"pred/{battery}_{dataset}_{model}.pkl"
    pred_old = pickle.load(open(fname("old"),"rb"))
    pred_new = pickle.load(open(fname("new"),"rb"))
    # Dunn index
    di_old = max([gdi(X, pred_old_K-1) for pred_old_K in pred_old.values()])
    di_new = max([gdi(X, pred_new_K-1) for pred_new_K in pred_new.values()])
    di_gain = di_new - di_old
    # Adjusted rand score
    ari_old = clustbench.get_score(y, pred_old, ari)
    ari_new = clustbench.get_score(y, pred_new, ari)
    ari_gain = ari_new - ari_old
    # Add additional measures to results
    results.loc[i, "di_gain"] = di_gain
    results.loc[i, "ari_old"] = ari_old
    results.loc[i, "ari_new"] = ari_new
    results.loc[i, "ari_gain"] = ari_gain

# Save the results
results = results.convert_dtypes()
results.to_csv(res_path+"results.csv", index=False)
