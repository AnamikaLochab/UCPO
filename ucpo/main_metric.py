import json
import os
import numpy as np


def pass_at_k(n, c, k):
    """
    Standard unbiased pass@k estimator.
    n: total number of samples
    c: number of correct samples
    k: pass@k
    """
    if n == 0:
        return np.nan
    if k > n:
        k = n
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def get_pass_at_n(res, n_list=None):
    if n_list is None:
        n_list = [1, 2, 4, 8, 16]

    dataset_passn = {}
    for dataset in res["acc"]:
        pass_at_n = {}
        for k in n_list:
            pass_rates = []
            for item in res["acc"][dataset]:
                n = len(item)
                c = sum(item)
                pass_rates.append(pass_at_k(n, c, k))
            pass_at_n[k] = float(np.nanmean(pass_rates) * 100.0) if len(pass_rates) > 0 else np.nan
        dataset_passn[dataset] = pass_at_n
    return dataset_passn


def compute_metric(metric_files, n_list=None):
    per_run_pass = {}
    per_run_length = {}

    for file in metric_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        pass_at_n = get_pass_at_n(data, n_list)
        print(f"Pass@k for {file}:")
        print(pass_at_n)

        per_run_pass[file] = pass_at_n
        per_run_length[file] = data.get("length", {})

    return per_run_pass, per_run_length


def safe_mean(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def fmt(v, digits=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.{digits}f}"


if __name__ == "__main__":
    metric_files = [ ]

    n_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    per_run_pass, per_run_length = compute_metric(metric_files, n_list=n_list)

    print("--- pass@k (per run) ---")
    print(per_run_pass)
    print("--- length (per run) ---")
    print(per_run_length)

    runs = list(per_run_pass.keys())
    datasets = sorted({ds for run_data in per_run_pass.values() for ds in run_data.keys()})

    output_tsv = "metrics_summary.tsv"
    with open(output_tsv, "w", encoding="utf-8") as f:
        header = ["run"]

        # dataset pass@k columns
        for ds in datasets:
            for k in n_list:
                header.append(f"{ds}@{k}")

        # dataset length columns
        for ds in datasets:
            header.append(f"{ds}_len")

        # average columns
        for k in n_list:
            header.append(f"avg@{k}")
        header.append("avg_len")

        f.write("\t".join(header) + "\n")

        for run in runs:
            run_name = os.path.splitext(os.path.basename(run))[0]
            row = [run_name]

            avg_pass_by_k = {k: [] for k in n_list}
            avg_len_vals = []

            # pass@k values
            for ds in datasets:
                for k in n_list:
                    v = per_run_pass.get(run, {}).get(ds, {}).get(k, np.nan)
                    row.append(fmt(v, digits=4))
                    if not np.isnan(v):
                        avg_pass_by_k[k].append(v)

            # length values
            for ds in datasets:
                v = per_run_length.get(run, {}).get(ds, np.nan)
                row.append(fmt(v, digits=2))
                if not (isinstance(v, float) and np.isnan(v)):
                    avg_len_vals.append(v)

            # averages
            for k in n_list:
                row.append(fmt(safe_mean(avg_pass_by_k[k]), digits=4))
            row.append(fmt(safe_mean(avg_len_vals), digits=4))

            f.write("\t".join(row) + "\n")

    print(f"Saved {output_tsv}")