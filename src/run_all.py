"""Run RQ1–RQ4 and write metrics, figures, and a machine-readable results JSON.

The processed CSV must already exist (python -m src.prepare_data).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.naive_bayes import MultinomialNB

from src.config import COUNT_MAX_FEATURES, DATA_DIR, FIG_DIR, MODELS, NB_M, RESULTS_DIR, SEED
from src.features import feature_summary, feature_table
from src.models import build_model
from src.plots import balance_heatmap, confusion, feature_boxes, grouped_bars, training_curves
from src.textutil import ABLATIONS, tokenize
from src.train import (
    TextSplit,
    build_vocab,
    make_loader,
    metric_bundle,
    predict,
    set_seed,
    train_model,
)

LABELS = list(MODELS)
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}


def _split(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    return frame[frame["split"] == name].copy()


def run_neural(frame: pd.DataFrame, arch: str, view: str, tag: str, response_col: str = "response") -> dict:
    print(f"\n=== {tag} ({arch}, {view}) ===", flush=True)
    set_seed(SEED)
    train_df = _split(frame, "train")
    val_df = _split(frame, "val")
    test_df = _split(frame, "test")
    vocab = build_vocab(train_df, view, response_col=response_col)
    device = torch.device("cpu")
    model = build_model(arch, len(vocab), len(LABELS))
    train_loader = make_loader(TextSplit(train_df, view, vocab, LABEL_TO_ID, response_col), shuffle=True)
    val_loader = make_loader(TextSplit(val_df, view, vocab, LABEL_TO_ID, response_col), shuffle=False)
    test_loader = make_loader(TextSplit(test_df, view, vocab, LABEL_TO_ID, response_col), shuffle=False)
    model, history = train_model(model, train_loader, val_loader, ID_TO_LABEL, device)
    y_true, y_pred = predict(model, test_loader, device)
    metrics = metric_bundle(y_true, y_pred, ID_TO_LABEL)
    by_domain = {}
    domains = test_df["domain"].to_numpy()
    for domain in sorted(set(domains)):
        mask = domains == domain
        if mask.sum() == 0:
            continue
        by_domain[domain] = metric_bundle(y_true[mask], y_pred[mask], ID_TO_LABEL)
    metrics["by_domain"] = by_domain
    metrics["history"] = history
    metrics["vocab_size"] = len(vocab)
    metrics["n_train"] = int(len(train_df))
    metrics["n_val"] = int(len(val_df))
    metrics["n_test"] = int(len(test_df))
    metrics["arch"] = arch
    metrics["view"] = view
    metrics["tag"] = tag
    print(
        f"  TEST acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} n={metrics['n_test']}",
        flush=True,
    )
    return metrics


def _text_series(frame: pd.DataFrame, view: str, response_col: str = "response") -> pd.Series:
    if view == "output":
        return frame[response_col].fillna("")
    if view == "input":
        return frame["instruction"].fillna("")
    if view == "both":
        return frame["instruction"].fillna("") + "\n" + frame[response_col].fillna("")
    raise ValueError(view)


def _count_matrix(train_text, test_text):
    """Each column is one word, the X_i from Naive Bayes in Chapter 1."""
    vectorizer = CountVectorizer(
        analyzer=tokenize,
        min_df=2,
        max_features=COUNT_MAX_FEATURES,
    )
    return vectorizer, vectorizer.fit_transform(train_text), vectorizer.transform(test_text)


def _score_predictions(test_df: pd.DataFrame, pred_labels, arch: str, view: str, n_train: int) -> dict:
    y_true = test_df["model"].map(LABEL_TO_ID).to_numpy()
    y_pred = pd.Series(pred_labels).map(LABEL_TO_ID).to_numpy()
    metrics = metric_bundle(y_true, y_pred, ID_TO_LABEL)
    metrics["arch"] = arch
    metrics["view"] = view
    metrics["n_train"] = int(n_train)
    metrics["n_test"] = int(len(test_df))
    print(f"  TEST acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}", flush=True)
    return metrics


def run_count_baselines(frame: pd.DataFrame, view: str, response_col: str = "response") -> dict:
    """Naive Bayes (MAP, m=1) and multi-class logistic regression on word counts."""
    print(f"\n=== count baselines {view} ===", flush=True)
    train_df = _split(frame, "train")
    test_df = _split(frame, "test")
    vectorizer, x_train, x_test = _count_matrix(
        _text_series(train_df, view, response_col),
        _text_series(test_df, view, response_col),
    )
    y_train = train_df["model"]
    nb = MultinomialNB(alpha=NB_M)
    nb.fit(x_train, y_train)
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=500)
    lr.fit(x_train, y_train)
    out = {
        "nb": _score_predictions(test_df, nb.predict(x_test), "naive_bayes", view, len(train_df)),
        "lr": _score_predictions(test_df, lr.predict(x_test), "logreg", view, len(train_df)),
    }
    if view == "output":
        # Chapter 1: the weight on a feature is how important that feature is.
        names = np.array(vectorizer.get_feature_names_out())
        top = {}
        for index, label in enumerate(lr.classes_):
            weights = lr.coef_[index]
            order = np.argsort(weights)
            top[str(label)] = {
                "high": [
                    {"word": str(names[j]), "weight": float(weights[j])}
                    for j in order[-12:][::-1]
                ],
                "low": [
                    {"word": str(names[j]), "weight": float(weights[j])}
                    for j in order[:8]
                ],
            }
        out["lr_top_words"] = top
        test_counts = x_test
        clusters = KMeans(n_clusters=len(LABELS), n_init=10, random_state=SEED)
        assigned = clusters.fit_predict(test_counts)
        gold = test_df["model"].to_numpy()
        # Purity: for each cluster, the most common gold label's share, then average by size.
        purity_hits = 0
        for cluster_id in range(len(LABELS)):
            members = gold[assigned == cluster_id]
            if len(members) == 0:
                continue
            purity_hits += int(pd.Series(members).value_counts().iloc[0])
        out["kmeans"] = {
            "k": len(LABELS),
            "nmi": float(normalized_mutual_info_score(gold, assigned)),
            "purity": purity_hits / max(len(gold), 1),
            "n_test": int(len(gold)),
        }
        print(
            f"  kmeans NMI={out['kmeans']['nmi']:.4f} purity={out['kmeans']['purity']:.4f}",
            flush=True,
        )
    return out


def per_model_recall(metrics: dict) -> dict:
    return {name: metrics["per_class"][name]["recall"] for name in metrics["labels"]}


def run_rq3_both_tests(frame: pd.DataFrame, arch: str, held_out: str) -> dict:
    """One training run, then score held-out-domain test rows and in-domain test rows."""
    print(f"\n=== RQ3 held-out={held_out} arch={arch} ===", flush=True)
    set_seed(SEED)
    train_df = frame[(frame["split"] == "train") & (frame["domain"] != held_out)].copy()
    val_df = frame[(frame["split"] == "val") & (frame["domain"] != held_out)].copy()
    cross_df = frame[(frame["split"] == "test") & (frame["domain"] == held_out)].copy()
    in_df = frame[(frame["split"] == "test") & (frame["domain"] != held_out)].copy()
    vocab = build_vocab(train_df, "output")
    device = torch.device("cpu")
    model = build_model(arch, len(vocab), len(LABELS))
    model, history = train_model(
        model,
        make_loader(TextSplit(train_df, "output", vocab, LABEL_TO_ID), True),
        make_loader(TextSplit(val_df, "output", vocab, LABEL_TO_ID), False),
        ID_TO_LABEL,
        device,
    )
    out = {"history": history, "arch": arch, "held_out_domain": held_out, "vocab_size": len(vocab)}
    for name, part in (("cross", cross_df), ("in_domain", in_df)):
        y_true, y_pred = predict(
            model, make_loader(TextSplit(part, "output", vocab, LABEL_TO_ID), False), device
        )
        metrics = metric_bundle(y_true, y_pred, ID_TO_LABEL)
        metrics["n"] = int(len(part))
        metrics["recall_by_model"] = per_model_recall(metrics)
        out[name] = metrics
        print(
            f"  {name}: acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} n={metrics['n']}",
            flush=True,
        )
    out["macro_f1_drop"] = out["in_domain"]["macro_f1"] - out["cross"]["macro_f1"]
    out["accuracy_drop"] = out["in_domain"]["accuracy"] - out["cross"]["accuracy"]
    return out


def round_metrics(obj):
    if isinstance(obj, dict):
        return {str(k): round_metrics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_metrics(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), 4)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    return obj


def main() -> None:
    set_seed(SEED)
    torch.set_num_threads(4)
    frame = pd.read_csv(DATA_DIR / "samples.csv.gz")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    counts = frame.groupby(["domain", "model"]).size().unstack()[list(MODELS)]
    balance_heatmap(counts, FIG_DIR / "domain_balance.png")

    results = {
        "seed": SEED,
        "n_rows": int(len(frame)),
        "split_counts": frame.groupby(["split", "model"]).size().unstack().to_dict(),
        "domain_counts": counts.to_dict(),
        "neural": {},
        "baselines": {},
        "rq3": {},
        "ablations": {},
        "extras": {},
    }

    # RQ1 and RQ2. Output-only is RQ1. The three views are RQ2.
    # "both" is the Chapter 7 concatenation of the instruction and the response.
    display = {"cnn": "CNN", "lstm": "LSTM"}
    curve_bundle = {}
    for view in ("output", "input", "both"):
        for arch in ("cnn", "lstm"):
            metrics = run_neural(frame, arch, view, tag=f"rq_{view}_{arch}")
            results["neural"][f"{arch}_{view}"] = metrics
            if view == "output":
                curve_bundle[display[arch]] = metrics["history"]
                confusion(
                    metrics["confusion_matrix"],
                    metrics["labels"],
                    f"{display[arch]} output-only test counts",
                    FIG_DIR / f"cm_{arch}_output.png",
                )
    training_curves(curve_bundle, FIG_DIR / "training_curves_output.png")

    for view in ("output", "input", "both"):
        results["baselines"][view] = run_count_baselines(frame, view)

    # Extras that the lectures do not cover, scored only on output text.
    for arch in ("bilstm", "cnn_multi"):
        results["extras"][arch] = run_neural(frame, arch, "output", tag=f"extra_{arch}")

    rq2_rows = []
    for view in ("input", "output", "both"):
        for arch, model_name in (("cnn", "CNN"), ("lstm", "LSTM")):
            block = results["neural"][f"{arch}_{view}"]
            rq2_rows.append(
                {"setting": view, "model": model_name, "score": block["macro_f1"], "accuracy": block["accuracy"]}
            )
        for key, model_name in (("nb", "Naive Bayes"), ("lr", "Logistic regression")):
            block = results["baselines"][view][key]
            rq2_rows.append(
                {"setting": view, "model": model_name, "score": block["macro_f1"], "accuracy": block["accuracy"]}
            )
    grouped_bars(rq2_rows, "RQ2: average F by text view", FIG_DIR / "rq2_views.png")
    results["rq2_plot_rows"] = rq2_rows

    # RQ3: writing held out, and code held out. Source task S, target task T (Ch6).
    for held in ("writing", "code"):
        for arch in ("cnn", "lstm"):
            results["rq3"][f"{arch}_{held}"] = run_rq3_both_tests(frame, arch, held)

    rq3_rows = []
    for held in ("writing", "code"):
        for arch in ("cnn", "lstm"):
            block = results["rq3"][f"{arch}_{held}"]
            rq3_rows.append(
                {"setting": f"in-domain (no {held})", "model": display[arch], "score": block["in_domain"]["macro_f1"], "held": held}
            )
            rq3_rows.append(
                {"setting": f"test on {held}", "model": display[arch], "score": block["cross"]["macro_f1"], "held": held}
            )
    # Two figures, one per held-out domain, so the bars stay readable.
    for held in ("writing", "code"):
        rows = [r for r in rq3_rows if r["held"] == held]
        grouped_bars(
            rows,
            f"RQ3 average F when {held} is held out of training",
            FIG_DIR / f"rq3_{held}.png",
        )

    # RQ4 surface features on the full balanced set (train+val+test), output text.
    feats = feature_table(frame)
    feats.to_csv(RESULTS_DIR / "surface_features.csv.gz", index=False, compression="gzip")
    results["rq4_features"] = feature_summary(feats)
    feature_boxes(feats, FIG_DIR / "rq4_features.png")

    # Ablations retrain on transformed outputs. Val/test use the same transform.
    ablation_rows = []
    base_scores = {
        "cnn": results["neural"]["cnn_output"]["macro_f1"],
        "lstm": results["neural"]["lstm_output"]["macro_f1"],
    }
    for arch in ("cnn", "lstm"):
        ablation_rows.append({"setting": "original", "model": display[arch], "score": base_scores[arch]})
    for ablation_name, fn in ABLATIONS.items():
        altered = frame.copy()
        altered["response_ablated"] = altered["response"].map(fn)
        for arch in ("cnn", "lstm"):
            metrics = run_neural(
                altered,
                arch,
                "output",
                tag=f"ablate_{ablation_name}_{arch}",
                response_col="response_ablated",
            )
            # The neural runner reads column via response_col for encoding, good.
            results["ablations"][f"{arch}_{ablation_name}"] = metrics
            ablation_rows.append(
                {"setting": ablation_name, "model": display[arch], "score": metrics["macro_f1"]}
            )
    grouped_bars(ablation_rows, "RQ4 ablations: output-only average F", FIG_DIR / "rq4_ablations.png")
    results["ablation_plot_rows"] = ablation_rows

    # Compact headline block for the report writer.
    headlines = {
        "rq1": {
            arch: {
                "accuracy": results["neural"][f"{arch}_output"]["accuracy"],
                "macro_f1": results["neural"][f"{arch}_output"]["macro_f1"],
                "per_class": results["neural"][f"{arch}_output"]["per_class"],
            }
            for arch in ("cnn", "lstm")
        },
        "rq1_baselines": {
            key: {
                "accuracy": results["baselines"]["output"][key]["accuracy"],
                "macro_f1": results["baselines"]["output"][key]["macro_f1"],
                "per_class": results["baselines"]["output"][key]["per_class"],
            }
            for key in ("nb", "lr")
        },
        "rq1_extras": {
            arch: {
                "accuracy": results["extras"][arch]["accuracy"],
                "macro_f1": results["extras"][arch]["macro_f1"],
            }
            for arch in ("bilstm", "cnn_multi")
        },
        "rq2_macro_f1": {
            **{
                f"{arch}_{view}": results["neural"][f"{arch}_{view}"]["macro_f1"]
                for arch in ("cnn", "lstm")
                for view in ("input", "output", "both")
            },
            **{
                f"{key}_{view}": results["baselines"][view][key]["macro_f1"]
                for key in ("nb", "lr")
                for view in ("input", "output", "both")
            },
        },
        "rq2_accuracy": {
            **{
                f"{arch}_{view}": results["neural"][f"{arch}_{view}"]["accuracy"]
                for arch in ("cnn", "lstm")
                for view in ("input", "output", "both")
            },
            **{
                f"{key}_{view}": results["baselines"][view][key]["accuracy"]
                for key in ("nb", "lr")
                for view in ("input", "output", "both")
            },
        },
        "rq3_drops": {
            key: {
                "in_macro_f1": block["in_domain"]["macro_f1"],
                "cross_macro_f1": block["cross"]["macro_f1"],
                "drop": block["macro_f1_drop"],
                "in_accuracy": block["in_domain"]["accuracy"],
                "cross_accuracy": block["cross"]["accuracy"],
                "cross_recall": block["cross"]["recall_by_model"],
                "in_recall": block["in_domain"]["recall_by_model"],
                "n_cross": block["cross"]["n"],
                "n_in": block["in_domain"]["n"],
            }
            for key, block in results["rq3"].items()
        },
    }
    results["headlines"] = headlines

    payload = round_metrics(results)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))
    print("\nWrote", RESULTS_DIR / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
