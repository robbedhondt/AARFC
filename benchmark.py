import numpy as np
from sklearn.base import ClusterMixin, BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import StratifiedKFold

class RandomForestClustering(ClusterMixin, BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=8, method="old", n_jobs=-1):
        self.n_clusters = n_clusters
        self.method = method
        self.n_jobs = n_jobs

    def fit(self, X):
        self.fit_forest(X)
        self.fit_affinity(X)
        self.fit_clustering(X)
        return self

    def fit_forest(self, X):
        if self.method == "old":
            self.rf = RandomForestClassifier(n_jobs=self.n_jobs)
            y_train = np.concatenate((np.ones(X.shape[0]), np.zeros(X.shape[0])))
            y_train = y_train.ravel()
            Xsynth = np.array(X, copy=True)
            for j in range(Xsynth.shape[1]):
                np.random.shuffle(Xsynth[:,j]) # shuffles in-place
            X_train = np.concatenate((X, Xsynth), axis=0)
        elif self.method == "new":
            self.rf = RandomForestRegressor(n_jobs=self.n_jobs)
            X_train = X
            y_train = StandardScaler().fit_transform(X).squeeze()
        self.rf.fit(X_train, y_train)
        return self
    
    def fit_affinity(self, X):
        paths, _ = self.rf.decision_path(X)
        scaling = np.array(paths.sum(axis=1)).flatten()
        affmat  = 2*(paths @ paths.T).toarray()
        affmat  = affmat / np.add.outer(scaling, scaling)
        self.affmat = affmat
        return self
    
    def fit_clustering(self, X):
        #print("Fitting the spectral clustering...")
        self.clus = SpectralClustering(n_clusters=self.n_clusters, affinity="precomputed")
        start = time.time()
        self.clus.fit(self.affmat)
        self.time_clus = time.time() - start
        self.labels_ = self.clus.labels_
        return self

if __name__ == "__main__":
    import os
    import clustbench
    import pandas as pd
    import pickle
    from tqdm import tqdm
    import time

    data_path = "clustering-data-v1-1.1.0/"
    res_path = "benchmark/"

    alg1 = RandomForestClustering(method="old", n_jobs=32)
    alg2 = RandomForestClustering(method="new", n_jobs=32)
    algorithms = {"old":alg1, "new":alg2}
    for battery in clustbench.get_battery_names(data_path):
        datasets = clustbench.get_dataset_names(battery, path=data_path)
        scoredf = pd.DataFrame(index=datasets, columns=["old","new"])
        for dataset in tqdm(datasets, desc=battery):
            benchmark = clustbench.load_dataset(battery, dataset, path=data_path)
            X = benchmark.data
            y = benchmark.labels
            if (battery == "mnist") or ((battery == "sipu") and (dataset in 
                                    ["birch1","birch2","worms_2","worms_64"])):
                # take a stratified sample of benchmark instead
                kfold = StratifiedKFold(n_splits=10) # 10% of the data
                ilabels = np.argmax(benchmark.n_clusters) # take labeling with most number of clusters
                iloc = list(kfold.split(X, y[ilabels]))[0][1]
                X = X[iloc,:]
                y = [labeling[iloc] for labeling in y]
            scores = [None, None]
            times  = [None, None]
            for i,(algname, alg) in enumerate(algorithms.items()):
                results_fname = res_path + f"pred/{battery}_{dataset}_{algname}.pkl"
                if os.path.exists(results_fname):
                    results = pickle.load(open(results_fname, 'rb'))
                else:
                    start = time.time()
                    # START: fit predict many
                    alg.fit_forest(X)
                    time_rf = time.time() - start
                    alg.fit_affinity(X)
                    time_aff = time.time() - start - time_rf
                    n_clusters = np.unique(np.r_[benchmark.n_clusters])
                    results = dict()
                    for k in n_clusters:
                        alg.n_clusters = int(k) # alg.set_params(n_clusters=k)
                        results[k] = alg.fit_clustering(X).labels_ + 1
                    time_clus = time.time() - start - time_rf - time_aff
                    # END: fit predict many
                    times[i] = time.time() - start
                    with open(res_path + "timing.txt", "a") as f:
                        f.write(f"{battery},{dataset},{algname},{times[i]},{time_rf},{time_aff},{time_clus}\n")
                    pickle.dump(results, open(results_fname, 'wb'))
                scores[i] = clustbench.get_score(y, results)
                scoredf.loc[benchmark.dataset, scoredf.columns[i]] = scores[i]
            # print(f"{battery}/{benchmark.dataset}: {scores[0]:.2f} <--> {scores[1]:.2f} (gain = {scores[1]-scores[0]:.2f})")
        scoredf["gain"] = scoredf["new"] - scoredf["old"]
        scoredf.to_csv(res_path + f"{battery}.csv")
