"""SVM RBF (scikit-learn) sobre a 1a derivada do espectro.

Pipeline: 1a diferença -> padronização por ponto -> SVC(kernel RBF, C=10).
Validação cruzada e ajuste usam só o treino; o teste é lido apenas no fim.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from core import load_csv, macro_f1, stratified_folds  # noqa: E402


def difference(X, order=1):
    return np.diff(X * 1000.0, n=order, axis=1)


def build_model(C, order=1):
    return make_pipeline(FunctionTransformer(difference, kw_args={"order": order}), StandardScaler(),
                         SVC(C=C, kernel="rbf", gamma="scale"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.environ.get("TRAIN_CSV", "/data/train_spectra.csv"))
    ap.add_argument("--test", default=os.environ.get("TEST_CSV", "/data/test_spectra.csv"))
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "/out"))
    ap.add_argument("--C", type=float, default=10.0)
    ap.add_argument("--diff", type=int, default=1, choices=range(1, 8), help="ordem da diferença do espectro")
    ap.add_argument("--cv-repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    start = time.time()
    ids, X, y = load_csv(args.train)
    print(f"[svm] treino: {X.shape[0]} espectros x {X.shape[1]} pontos, classes {np.bincount(y).tolist()}")
    metrics = {"model": "svm", "config": {"C": args.C, "features": f"{args.diff}a diferenca padronizada", "kernel": "rbf"}}

    if not args.skip_cv:
        f1s, accs, oof = [], [], np.zeros(len(y))
        for rep in range(args.cv_repeats):   # mesmos folds dos outros experimentos
            rng = np.random.default_rng(args.seed + rep)
            for k, val in enumerate(stratified_folds(y, 5, rng)):
                tr = np.setdiff1d(np.arange(len(y)), val)
                model = build_model(args.C, args.diff).fit(X[tr], y[tr])
                score = model.decision_function(X[val])
                oof[val] += score / args.cv_repeats
                pred = (score > 0).astype(int)
                f1s.append(macro_f1(y[val], pred)); accs.append(float(np.mean(pred == y[val])))
                print(f"  rep {rep + 1} fold {k + 1}: macro-F1 {f1s[-1]:.4f}  acc {accs[-1]:.4f}")
        metrics["cv"] = {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                         "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
                         "macro_f1_per_fold": f1s}
        print(f"[svm] CV macro-F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | acc = {np.mean(accs):.4f}")
        np.savetxt(os.path.join(args.out, "oof_predictions.csv"), np.column_stack([ids, y, oof]),
                   delimiter=",", header="id,class,svm_score", comments="", fmt=["%d", "%d", "%.6f"])

    model = build_model(args.C, args.diff).fit(X, y)
    metrics["n_support_vectors"] = int(model[-1].n_support_.sum())
    if os.path.exists(args.test):
        test_ids, X_test, _ = load_csv(args.test)
        pred = model.predict(X_test).astype(int)
        with open(os.path.join(args.out, "submission.csv"), "w") as f:
            f.write("id,class\n")
            f.writelines(f"{i},{c}\n" for i, c in zip(test_ids, pred))
        metrics["test_predicted_counts"] = np.bincount(pred, minlength=2).tolist()
        print(f"[svm] submissão gravada ({len(pred)} linhas, classes previstas {metrics['test_predicted_counts']})")
    metrics["seconds"] = round(time.time() - start, 1)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[svm] pronto em {metrics['seconds']}s")


if __name__ == "__main__":
    main()
