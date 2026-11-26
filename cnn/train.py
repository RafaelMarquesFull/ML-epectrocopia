"""CNN 1D (PyTorch) para detecção de diabetes a partir de espectros.

Mesma regra dos outros experimentos: validação cruzada e ajuste usam só o
treino; o teste é lido apenas no fim, para gerar a submissão.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from core import load_csv, macro_f1, stratified_folds  # noqa: E402


def make_inputs(X, mode, stats=None):
    """Monta os canais de entrada. stats vem sempre do treino."""
    X = X * 1000.0
    D = np.diff(X, axis=1, prepend=X[:, :1])
    if mode == "global":      # preserva amplitude e nível de ruído de cada espectro
        if stats is None:
            stats = (X.std(), D.std())
        chans = [X / stats[0], D / stats[1]]
    else:                     # "feature": padroniza cada ponto do espectro
        if stats is None:
            stats = (X.mean(0), X.std(0) + 1e-9, D.mean(0), D.std(0) + 1e-9)
        chans = [(X - stats[0]) / stats[1], (D - stats[2]) / stats[3]]
    return torch.tensor(np.stack(chans, 1), dtype=torch.float32), stats


class SpectraCNN(nn.Module):
    def __init__(self, width=24, kernel=9, dropout=0.3):
        super().__init__()
        chans = [2, width, width * 2, width * 4, width * 4]
        layers = []
        for i in range(4):
            layers += [nn.Conv1d(chans[i], chans[i + 1], kernel, stride=2, padding=kernel // 2),
                       nn.BatchNorm1d(chans[i + 1]), nn.GELU()]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(chans[-1] * 2, 1))

    def forward(self, x):
        h = self.features(x)
        return self.head(torch.cat([h.mean(2), h.amax(2)], 1)).squeeze(1)


def train_cnn(Xt, y, cfg, seed):
    torch.manual_seed(seed)
    model = SpectraCNN(cfg["width"], cfg["kernel"], cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    n = len(Xt)
    steps = cfg["epochs"] * int(np.ceil(n / cfg["batch_size"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps)
    yt = torch.tensor(y, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(cfg["epochs"]):
        order = torch.randperm(n, generator=gen)
        for start in range(0, n, cfg["batch_size"]):
            idx = order[start:start + cfg["batch_size"]]
            if len(idx) < 2:
                continue
            xb = Xt[idx]
            if cfg["shift"]:   # aumento de dados: deslocamento de poucos pontos
                xb = torch.roll(xb, int(torch.randint(-cfg["shift"], cfg["shift"] + 1, (1,), generator=gen)), 2)
            opt.zero_grad()
            loss_fn(model(xb), yt[idx]).backward()
            opt.step()
            sched.step()
    return model.eval()


@torch.no_grad()
def predict(model, Xt):
    return torch.sigmoid(model(Xt)).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=os.environ.get("TRAIN_CSV", "/data/train_spectra.csv"))
    ap.add_argument("--test", default=os.environ.get("TEST_CSV", "/data/test_spectra.csv"))
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "/out"))
    ap.add_argument("--inputs", default="feature", choices=["global", "feature"])
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--kernel", type=int, default=9)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--shift", type=int, default=2)
    ap.add_argument("--cv-repeats", type=int, default=2)
    ap.add_argument("--ensemble", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--skip-final", action="store_true")
    args = ap.parse_args()
    cfg = {k: v for k, v in vars(args).items() if k not in ("train", "test", "out")}
    os.makedirs(args.out, exist_ok=True)
    start = time.time()
    ids, X, y = load_csv(args.train)
    print(f"[cnn] treino: {X.shape}, classes {np.bincount(y).tolist()}, config {cfg}", flush=True)
    metrics = {"model": "cnn", "config": cfg}

    if not args.skip_cv:
        f1s, accs, oof = [], [], np.zeros(len(y))
        for rep in range(args.cv_repeats):
            rng = np.random.default_rng(args.seed + rep)
            for k, val in enumerate(stratified_folds(y, 5, rng)):
                tr = np.setdiff1d(np.arange(len(y)), val)
                Xtr, stats = make_inputs(X[tr], args.inputs)
                model = train_cnn(Xtr, y[tr], cfg, args.seed + 1000 * rep + k)
                p = predict(model, make_inputs(X[val], args.inputs, stats)[0])
                oof[val] += p / args.cv_repeats
                pred = (p >= 0.5).astype(int)
                f1s.append(macro_f1(y[val], pred)); accs.append(float(np.mean(pred == y[val])))
                print(f"  rep {rep + 1} fold {k + 1}: macro-F1 {f1s[-1]:.4f} ({time.time() - start:.0f}s)", flush=True)
        metrics["cv"] = {"macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
                         "accuracy_mean": float(np.mean(accs)), "macro_f1_per_fold": f1s}
        print(f"[cnn] CV macro-F1 = {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | acc = {np.mean(accs):.4f}", flush=True)
        np.savetxt(os.path.join(args.out, "oof_predictions.csv"), np.column_stack([ids, y, oof]),
                   delimiter=",", header="id,class,prob_diabetes", comments="", fmt=["%d", "%d", "%.6f"])

    if not args.skip_final and os.path.exists(args.test):
        Xall, stats = make_inputs(X, args.inputs)
        models = [train_cnn(Xall, y, cfg, args.seed + 7 * i) for i in range(args.ensemble)]
        test_ids, X_test, _ = load_csv(args.test)
        Xte = make_inputs(X_test, args.inputs, stats)[0]
        pred = (np.mean([predict(m, Xte) for m in models], 0) >= 0.5).astype(int)
        with open(os.path.join(args.out, "submission.csv"), "w") as f:
            f.write("id,class\n")
            f.writelines(f"{i},{c}\n" for i, c in zip(test_ids, pred))
        metrics["test_predicted_counts"] = np.bincount(pred, minlength=2).tolist()
        metrics["n_params"] = sum(p.numel() for p in models[0].parameters())
        print(f"[cnn] submissão gravada, classes previstas {metrics['test_predicted_counts']}", flush=True)

    metrics["seconds"] = round(time.time() - start, 1)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[cnn] pronto em {metrics['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
