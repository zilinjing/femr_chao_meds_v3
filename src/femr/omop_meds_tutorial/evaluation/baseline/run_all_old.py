from train_utils import *

def load_features_labels(feature_path, label_path, split_path):
    data = pickle.load(open(feature_path, "rb"))
    labels = (
        pd.read_parquet(label_path)
        .set_index(["subject_id", "prediction_time"])
    )
    # print("loaded labels")

    X = pd.DataFrame(
        data["features"].todense(),
        index=pd.MultiIndex.from_arrays(
            [data["subject_ids"], data["feature_times"]],
            names=["subject_id", "prediction_time"],
        ),
    )

    # print("loaded X")
    X = X.loc[labels.index]
    X, labels = remove_duplicates(X, labels)

    split = pd.read_csv(split_path, index_col = 'subject_id')
    labels = labels.join(split)
    # print("loded split")

    # Split train set
    val = labels[labels.split_name == 'train'].index.to_series().sample(frac=0.2, replace=False, random_state=42)
    hyper = val.sample(frac=0.5, replace=False, random_state=42)
    # labels.split_name.loc[val] = 'tuning'
    # labels.split_name.loc[hyper] = 'hyper'
    labels.loc[val, "split_name"] = 'tuning'
    labels.loc[hyper, "split_name"] = 'hyper'
    print("finish load features")
    return X, labels

EXPERIMENTS = {
    "mimic": {
        "data_path": "/data2/processed_datasets/mimic/baseline_features_22_12_vj2292/",
        "tte_col": "time_to_event_days",

        "phenotypes": [
            "ami",
            "stroke",
            "masld",
        ],

        "binary_only": [
            "in_hospital_mortality",
            "readmission",
            "long_los",
        ],
    },

    # "cuimc": {
    #     "data_path": "/data2/processed_datasets/ehr_foundation_data/ohdsi_cumc_deid/"
    #                  "ohdsi_cumc_deid_2023q4r3_v3_mapped/baseline_features_22_12_vj2292/",
    #     "tte_col": "time_to_event",

    #     "phenotypes": [
    #         "AMI",
    #         "Ischemic_Stroke",
    #         "MASLD",
    #         "Celiac",
    #     ],

    #     "binary_only": [
    #         "death",
    #         "readmission",
    #         "long_los",
    #     ],
    # },
}

for dataset, cfg in EXPERIMENTS.items():
    # results_path = f"/data/results/vj2292/baselines/{dataset}"
    split_path = "/data2/processed_datasets/zj2398/femr/{}/main_split.csv".format("cumc" if dataset == "cuimc" else dataset)
    # if dataset == 'mimic': continue
    # ------------------------
    # Phenotypes: survival + binary
    # ------------------------
    for name in cfg["phenotypes"]:
        results_path = f"/data/results/zj2398/femr/{dataset}/classification/xgboost"
        for outcome, suffix, model in [
            ("survival", "", "deephit"),
            ("binary", "_binary", "xgboost"),
        ]:
            out_file = os.path.join(results_path, f"{name}_{model}.pickle")

            print("-" * 42)
            print(dataset, name, f"[{outcome}]")

            if os.path.exists(out_file):
                summarize(load_metrics(out_file))
                continue

            try:
                base_dir = os.path.join(cfg["data_path"], f"{name}{suffix}")
                if outcome == "survival":
                    print("continue skip survival")
                    continue
                X, y = load_features_labels(
                    os.path.join(base_dir, "features", f"{name}.pkl"),
                    os.path.join(base_dir, "labels",   f"{name}.parquet"),
                    split_path
                )

                if outcome == "survival":
                    splits = split_survival(X, y, cfg["tte_col"])
                    metrics = train_survival(*splits, model_type="deephit")
                else:
                    splits = split_binary(X, y)
                    # metrics = train_binary(*splits)
                    print("begin training")
                    metrics, pred_df = train_binary(*splits)
                    pred_df.to_parquet(out_file.replace(".pickle", "_predictions.parquet"), index=False)

                save_metrics(metrics, out_file)

            except Exception as e:
                print(f"[ERROR] {dataset} | {name} | {outcome}: {e}")

    # ------------------------
    # Binary-only outcomes
    # ------------------------
    for name in cfg["binary_only"]:
        results_path = f"/data/results/zj2398/femr/{dataset}/phenotype/deephit"
        out_file = os.path.join(results_path, f"{name}_xgboost.pickle")

        print("-" * 42)
        print(dataset, name, "[binary-only]")

        if os.path.exists(out_file):
            summarize(load_metrics(out_file))
            continue

        try:
            base_dir = os.path.join(cfg["data_path"], f"{name}_binary")

            X, y = load_features_labels(
                os.path.join(base_dir, "features", f"{name}.pkl"),
                os.path.join(base_dir, "labels",   f"{name}.parquet"),
                split_path
            )

            splits = split_binary(X, y)
            # metrics = train_binary(*splits)
            metrics, pred_df = train_binary(*splits)
            pred_df.to_parquet(out_file.replace(".pickle", "_predictions.parquet"), index=False)
            save_metrics(metrics, out_file)

        except Exception as e:
            print(f"[ERROR] {dataset} | {name} | binary-only: {e}")