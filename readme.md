# Autoassociative Random Forest Clustering
This repository hosts the code used in the publication "Autoassociative Random Forest Clustering" by Robbe D'hondt, Felipe Kenji Nakano, and Celine Vens. This publication was accepted into the 12th workshop on New Frontiers in Mining Complex Patterns (https://nfmcp2024.di.uniba.it/) and presented at the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases in Vilnius, Lithuania.

To set up your environment:
```
python3.11 -m pip install -r requirements.txt
```

Download the data from https://github.com/gagolews/clustering-data-v1/releases/tag/v1.1.0 .

To run the benchmark experiments for both methods:
```
mkdir benchmark
mkdir benchmark/pred
touch benchmark/timing.txt
python3.11 benchmark.py
```

To post-process the results into a dataframe:
```
python3.11 results.py
```
The dataframe is then saved to `benchmark/results.csv`. 
Our version of this file is included with this repository.
