import argparse
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

from ca_plus import ClusterCAplus
from TorqueClustering_weighted import TorqueClustering_weighted
from utils.eval_metrics import clustering_evaluation_metrics


DEFAULT_OPENML_DATASETS = {
    "optdigits": 28,
}


def load_openml_dataset(dataset_name: str, data_id: int | None = None):
    """Load a numeric classification dataset from OpenML."""
    if data_id is None:
        data_id = DEFAULT_OPENML_DATASETS.get(dataset_name.lower())

    if data_id is not None:
        data, target = fetch_openml(data_id=data_id, return_X_y=True, as_frame=False)
    else:
        data, target = fetch_openml(name=dataset_name, version=1, return_X_y=True, as_frame=False)

    data = np.asarray(data, dtype=np.float64)
    target = LabelEncoder().fit_transform(target).astype(int)
    return data, target, int(np.unique(target).size)


def run_single_trial(
    trial_idx: int,
    data: np.ndarray,
    target: np.ndarray,
    mode: str,
    cluster_selection: str,
    band: float,
):
    rng = np.random.RandomState(trial_idx)
    x_train, y_true = shuffle(data, target, random_state=rng)

    model = ClusterCAplus()
    ca_start = time.time()

    if mode == "stationary":
        model.fit(x_train)
    elif mode == "nonstationary":
        for current_class in rng.permutation(np.unique(y_true)):
            class_idx = np.where(y_true == current_class)[0]
            model.fit(x_train[class_idx])
    else:
        raise ValueError("mode must be 'stationary' or 'nonstationary'.")

    ca_time = time.time() - ca_start

    labels_ca = model.predict(x_train)
    ami_ca, ari_ca = clustering_evaluation_metrics(y_true, labels_ca)
    num_nodes = len(model.G_.nodes())
    num_clusters_ca = int(np.unique(labels_ca).size)

    node_ids = sorted(model.G_.nodes())
    node_centers = np.array([model.G_.nodes[node_id]["weight"] for node_id in node_ids])
    winning_counts = nx.get_node_attributes(model.G_, "winning_counts")
    node_masses = np.array([winning_counts.get(node_id, 1) for node_id in node_ids], dtype=float)
    dm_nodes = cdist(node_centers, node_centers, metric="euclidean")

    tc_start = time.time()
    labels_nodes_tc, *_ = TorqueClustering_weighted(
        dm_nodes,
        node_masses=node_masses,
        K=0,
        isnoise=False,
        isfig=False,
        matlab_compatibility=True,
        use_std_adjustment=True,
        adjustment_factor=0.5,
        cluster_selection=cluster_selection,
        num_band=band,
    )
    tc_time = time.time() - tc_start

    nearest_center_idx = np.argmin(cdist(x_train, node_centers, metric="euclidean"), axis=1)
    labels_tc = labels_nodes_tc[nearest_center_idx]
    ami_tc, ari_tc = clustering_evaluation_metrics(y_true, labels_tc)

    return {
        "trial": trial_idx,
        "band": band,
        "CAplus_AMI": ami_ca,
        "CAplus_ARI": ari_ca,
        "CAplus_time_sec": ca_time,
        "CAplus_num_nodes": num_nodes,
        "CAplus_num_clusters": num_clusters_ca,
        "CAplusTC_AMI": ami_tc,
        "CAplusTC_ARI": ari_tc,
        "CAplusTC_time_sec": ca_time + tc_time,
        "CAplusTC_num_clusters": int(np.unique(labels_tc).size),
    }


def summarize_results(rows: list[dict], dataset: str, mode: str, num_classes: int) -> pd.DataFrame:
    metrics = [
        ("AMI", "AMI", 4),
        ("ARI", "ARI", 4),
        ("time_sec", "time_sec", 4),
    ]
    summary_rows = []
    df = pd.DataFrame(rows)

    for method, prefix in [("CAplus", "CAplus"), ("CAplus+TC", "CAplusTC")]:
        out = {
            "method": method,
            "dataset": dataset,
            "mode": mode,
            "#classes": num_classes,
            "num_trials": len(df),
        }
        if method == "CAplus":
            out["nodes"] = _fmt(df["CAplus_num_nodes"], 2)
            out["clusters"] = _fmt(df["CAplus_num_clusters"], 2)
        else:
            out["nodes"] = _fmt(df["CAplus_num_nodes"], 2)
            out["clusters"] = _fmt(df["CAplusTC_num_clusters"], 2)

        for column_suffix, output_name, digits in metrics:
            out[output_name] = _fmt(df[f"{prefix}_{column_suffix}"], digits)
        summary_rows.append(out)

    return pd.DataFrame(summary_rows)


def _fmt(values: pd.Series, digits: int) -> str:
    return f"{values.mean():.{digits}f}+/-{values.std(ddof=0):.{digits}f}"


def parse_args():
    parser = argparse.ArgumentParser(description="Run CA+TC on an OpenML dataset.")
    parser.add_argument("--dataset", default="optdigits", help="OpenML dataset name.")
    parser.add_argument("--data-id", type=int, default=28, help="OpenML data_id. Default: OptDigits (28).")
    parser.add_argument("--mode", choices=["stationary", "nonstationary"], default="stationary")
    parser.add_argument("--cluster-selection", choices=["paper_eq4", "gap", "quantile"], default="quantile")
    parser.add_argument("--band", type=float, default=0.1, help="Quantile band used when cluster-selection=quantile.")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials. Default: 1.")
    parser.add_argument("--output-dir", default="results_CAplus_TC")
    return parser.parse_args()


def main():
    args = parse_args()
    data, target, num_classes = load_openml_dataset(args.dataset, args.data_id)

    rows = [
        run_single_trial(
            trial_idx=trial,
            data=data,
            target=target,
            mode=args.mode,
            cluster_selection=args.cluster_selection,
            band=args.band,
        )
        for trial in range(args.trials)
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trial_df = pd.DataFrame(rows)
    summary_df = summarize_results(rows, args.dataset, args.mode, num_classes)

    band_label = f"{args.band:g}"
    trial_path = output_dir / f"{args.dataset}_trials_band{band_label}.csv"
    summary_path = output_dir / f"{args.dataset}_summary_band{band_label}.csv"
    trial_df.to_csv(trial_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("============================================")
    print(f"Dataset: {args.dataset} (OpenML data_id={args.data_id})")
    print(f"Samples: {len(data)}, Features: {data.shape[1]}, Classes: {num_classes}")
    print(summary_df.to_string(index=False))
    print("--------------------------------------------")
    print(f"Saved trial results: {trial_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
