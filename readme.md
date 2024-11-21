# Autoassociative Random Forest Clustering
This repository hosts the code used in the publication "Autoassociative Random Forest Clustering" by Robbe D'hondt, Felipe Kenji Nakano, and Celine Vens. 
This publication was accepted into the 12th workshop on _[New Frontiers in Mining Complex Patterns](https://nfmcp2024.di.uniba.it/)_ (NFMCP) and presented at the _European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases_ (ECML-PKDD) in Vilnius, Lithuania.

## How to cite us
The paper will be included in a joint post-workshop proceeding published by _Springer Communications in Computer and Information Science_ (CCIS).
There is no published paper yet, but the bib reference will look something like this:
```
@inproceedings{dhondt2024autoassociative,
  title={Autoassociative Random Forest Clustering},
  author={D’hondt, Robbe and Nakano, Felipe Kenji and Vens, Celine},
  year={2024},
  maintitle={European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases},
  booktitle={12th workshop on New Frontiers in Mining Complex Patterns},
  url={https://nfmcp2024.di.uniba.it/papers/paper226.pdf}
}
```

## Running the experiments yourself
To set up your environment:
```
python3.11 -m pip install -r requirements.txt
```

Download the data from https://github.com/gagolews/clustering-data-v1/releases/tag/v1.1.0 .

To run the benchmark experiments for both methods:
```
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
