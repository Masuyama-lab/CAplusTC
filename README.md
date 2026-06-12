# CA+TC

(c) 2026 Naoki Masuyama

CA+TC is a clustering method that first learns CA+ nodes from input samples and
then merges those nodes with Torque Clustering (TC). The method is designed to
clarify the global cluster structure of CA+ by using mass- and distance-guided
cluster merging.

## Requirements

- Python 3.10 or later
- numpy
- pandas
- scipy
- scikit-learn
- networkx
- matplotlib

Install the required packages with your preferred Python environment, for
example:

```bash
pip install -r requirements.txt
```

## Main Files

- `main.py`: example runner for CA+ and CA+TC on OptDigits.
- `ca_plus.py`: core CA+ model.
- `TorqueClustering_weighted.py`: weighted Torque Clustering for CA+ nodes.
- `utils/`: utility functions for TC, evaluation metrics, and dataset helpers.

## Data

`main.py` downloads OptDigits from OpenML through
`sklearn.datasets.fetch_openml`. The default dataset is OptDigits
(`OpenML data_id=28`).

The runner expects numeric features and classification labels. To use another
OpenML dataset, pass `--dataset` and, if available, `--data-id`.

## Run

```bash
python main.py
```

The script prints the number of samples, features, classes, nodes, clusters,
runtime, ARI, and AMI. By default, it runs 1 trial.

When `python main.py` is executed, the runner:

1. Downloads OptDigits from OpenML (`data_id=28`).
2. Shuffles the samples for each trial.
3. Trains CA+ on the shuffled data.
4. Applies weighted Torque Clustering to the learned CA+ nodes.
5. Evaluates CA+ and CA+TC with AMI and ARI.
6. Writes trial-level and summary CSV files to `results_CAplus_TC/`.

Useful options:

```bash
python main.py --trials 3
python main.py --mode nonstationary
python main.py --cluster-selection quantile --band 0.1
```

## Citation

If you use this code, please cite:

S. Inoue, N. Masuyama, Y. Nojima, Y. Toda, Z. Liu, C. K. Loo, and W. S. Liew,
"Cluster merging in adaptive resonance theory-based clustering guided by mass
and distance criteria," in Proc. of the 2026 International Joint Conference on
Neural Networks (IJCNN), Maastricht, Netherlands, pp. 1-6, June 21-26, 2026.

## Third-party Code and Licenses

This repository contains both original code and code adapted from third-party
implementations. Do not assume that the whole repository is available under a
single permissive license.

The Torque Clustering-related files reuse or adapt code from the Torque
Clustering implementation by Jie Yang:
https://github.com/Cognet-74/TorqueClusteringPy

The adapted Torque Clustering files include a Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
notice. Commercial use may be restricted by that license. The following files
are treated as reused or adapted from that repository:

- `TorqueClustering_weighted.py`
- `utils/Final_label.py`
- `utils/Nab_dec.py`
- `utils/Nab_dec_quantile.py`
- `utils/Updateljmat_weighted.py`
- `utils/dataset_config.py`
- `utils/mindisttwinsloc.py`
- `utils/ps2psdist.py`
- `utils/uniqueZ.py`

See `THIRD_PARTY_NOTICES.md` for the current provenance notes. Before releasing
or redistributing this repository, review each file and upstream dependency so
that the repository-level license, notices, and redistribution terms are
consistent with the included code.
