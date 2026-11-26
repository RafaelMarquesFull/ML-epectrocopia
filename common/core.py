"""MLP em NumPy puro para detecção de diabetes a partir de espectros.

Tudo que é compartilhado entre os três experimentos (leaky_relu, gelu, swiglu)
fica aqui; a única coisa que muda entre eles é a ativação das camadas ocultas.

Regra da disciplina respeitada pelo código: o conjunto de teste nunca entra em
ajuste nenhum. Pré-processamento, validação cruzada e treino usam só o treino;
o teste é lido apenas no fim, para gerar o arquivo de submissão.
"""
import argparse
import json
import os
import time

import numpy as np

ACTIVATIONS = ("leaky_relu", "gelu", "swiglu")

# Mesmos hiperparâmetros para as três ativações, para a comparação ser justa.
CONFIG = {
    "snv": True,            # normaliza cada espectro (média 0, desvio 1)
    "n_components": 80,     # PCA ajustado só no treino
    "hidden": [64, 32],
    "dropout": 0.2,
    "input_noise": 0.1,     # ruído gaussiano nas entradas durante o treino
    "lr": 2e-3,
    "weight_decay": 1e-2,
    "epochs": 300,
    "batch_size": 32,
    "leaky_slope": 0.01,
    "cv_folds": 5,
    "cv_repeats": 3,
    "ensemble": 5,          # modelos (sementes) no treino final
    "seed": 42,
}


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
def load_csv(path):
    """Devolve (ids, X, y). y é None quando o arquivo não tem a coluna class."""
    with open(path) as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    ids = data[:, 0].astype(int)
    if header[-1] == "class":
        return ids, data[:, 1:-1], data[:, -1].astype(int)
    return ids, data[:, 1:], None


class Preprocessor:
    """SNV por espectro -> padronização por feature -> PCA -> escala unitária.

    fit() só deve receber dados de treino.
    """

    def __init__(self, n_components, snv=True):
        self.n_components = n_components
        self.snv = snv

    def _snv(self, X):
        if not self.snv:
            return X
        return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)

    def fit(self, X):
        X = self._snv(X)
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-12
        Z = (X - self.mean) / self.std
        if self.n_components:
            _, S, Vt = np.linalg.svd(Z, full_matrices=False)
            k = min(self.n_components, len(S))
            self.components = Vt[:k].T
            self.scale = S[:k] / np.sqrt(len(X) - 1)
        return self

    def transform(self, X):
        Z = (self._snv(X) - self.mean) / self.std
        if self.n_components:
            Z = Z @ self.components / self.scale
        return Z


def stratified_folds(y, k, rng):
    """Lista de k arrays de índices de validação, com proporção de classes mantida."""
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = rng.permutation(np.where(y == cls)[0])
        for i, part in enumerate(np.array_split(idx, k)):
            folds[i].extend(part)
    return [np.array(sorted(f)) for f in folds]


def macro_f1(y_true, y_pred):
    scores = []
    for cls in (0, 1):
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(scores))


# --------------------------------------------------------------------------
# Ativações
# --------------------------------------------------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def swish(x):
    s = sigmoid(x)
    return x * s, s + x * s * (1 - s)


def leaky_relu(x, slope):
    return np.where(x > 0, x, slope * x), np.where(x > 0, 1.0, slope)


_GELU_C = np.sqrt(2.0 / np.pi)


def gelu(x):
    """GELU na aproximação tanh (a mesma do BERT/GPT-2); NumPy não tem erf."""
    u = _GELU_C * (x + 0.044715 * x**3)
    t = np.tanh(u)
    du = _GELU_C * (1 + 3 * 0.044715 * x**2)
    return 0.5 * x * (1 + t), 0.5 * (1 + t) + 0.5 * x * (1 - t**2) * du


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
class MLP:
    """Camadas ocultas com a ativação escolhida e saída linear (1 logit).

    leaky_relu / gelu: h = f(xW + b)
    swiglu:            h = swish(xW + b) * (xV + c)
    No SwiGLU a largura é 2/3 da configurada, para o número de parâmetros
    ficar próximo ao das outras duas (mesma convenção dos transformers).
    """

    def __init__(self, n_in, hidden, activation, rng, dropout=0.0, leaky_slope=0.01):
        assert activation in ACTIVATIONS
        self.activation = activation
        self.dropout = dropout
        self.leaky_slope = leaky_slope
        self.params = []   # lista plana de arrays
        self.layers = []   # índices em self.params por camada
        gated = activation == "swiglu"
        prev = n_in
        for h in hidden:
            h = max(2, round(h * 2 / 3)) if gated else h
            idx = []
            for _ in range(2 if gated else 1):
                idx.append(len(self.params))
                self.params.append(rng.normal(0, np.sqrt(2.0 / prev), (prev, h)))
                self.params.append(np.zeros(h))
            self.layers.append(idx)
            prev = h
        self.out = len(self.params)
        self.params.append(rng.normal(0, np.sqrt(1.0 / prev), (prev, 1)))
        self.params.append(np.zeros(1))

    def n_params(self):
        return int(sum(p.size for p in self.params))

    def forward(self, X, rng=None):
        """rng != None liga o dropout (modo treino). Devolve logits e o cache."""
        P = self.params
        cache = []
        h = X
        for idx in self.layers:
            a = h @ P[idx[0]] + P[idx[0] + 1]
            if self.activation == "swiglu":
                g = h @ P[idx[1]] + P[idx[1] + 1]
                s, ds = swish(a)
                out = s * g
                c = {"x": h, "g": g, "s": s, "ds": ds}
            else:
                out, d = leaky_relu(a, self.leaky_slope) if self.activation == "leaky_relu" else gelu(a)
                c = {"x": h, "d": d}
            if rng is not None and self.dropout:
                mask = (rng.random(out.shape) >= self.dropout) / (1 - self.dropout)
                out = out * mask
                c["mask"] = mask
            cache.append(c)
            h = out
        logits = (h @ P[self.out] + P[self.out + 1]).ravel()
        return logits, (cache, h)

    def backward(self, dlogits, cache):
        """dlogits = dLoss/dlogits. Devolve gradientes na ordem de self.params."""
        cache, h_last = cache
        grads = [None] * len(self.params)
        d = dlogits[:, None]
        grads[self.out] = h_last.T @ d
        grads[self.out + 1] = d.sum(0)
        d = d @ self.params[self.out].T
        for idx, c in zip(reversed(self.layers), reversed(cache)):
            if "mask" in c:
                d = d * c["mask"]
            if self.activation == "swiglu":
                da = d * c["g"] * c["ds"]
                dg = d * c["s"]
                grads[idx[1]] = c["x"].T @ dg
                grads[idx[1] + 1] = dg.sum(0)
                back = da @ self.params[idx[0]].T + dg @ self.params[idx[1]].T
            else:
                da = d * c["d"]
                back = da @ self.params[idx[0]].T
            grads[idx[0]] = c["x"].T @ da
            grads[idx[0] + 1] = da.sum(0)
            d = back
        return grads

    def predict_proba(self, X):
        return sigmoid(self.forward(X)[0])


def bce_with_logits(logits, y):
    """Perda média e seu gradiente em relação aos logits."""
    loss = np.mean(np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits))))
    return loss, (sigmoid(logits) - y) / len(y)


def train_model(X, y, activation, cfg, seed):
    """AdamW com decaimento cosseno da taxa de aprendizado, épocas fixas."""
    rng = np.random.default_rng(seed)
    model = MLP(X.shape[1], cfg["hidden"], activation, rng, cfg["dropout"], cfg["leaky_slope"])
    m = [np.zeros_like(p) for p in model.params]
    v = [np.zeros_like(p) for p in model.params]
    b1, b2, eps = 0.9, 0.999, 1e-8
    n, bs = len(X), cfg["batch_size"]
    steps = cfg["epochs"] * int(np.ceil(n / bs))
    t = 0
    for _ in range(cfg["epochs"]):
        order = rng.permutation(n)
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            xb = X[idx]
            if cfg["input_noise"]:
                xb = xb + rng.normal(0, cfg["input_noise"], xb.shape)
            logits, cache = model.forward(xb, rng)
            _, dlogits = bce_with_logits(logits, y[idx])
            grads = model.backward(dlogits, cache)
            lr = cfg["lr"] * 0.5 * (1 + np.cos(np.pi * t / steps))
            t += 1
            for i, (p, g) in enumerate(zip(model.params, grads)):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                if p.ndim == 2:  # weight decay só nos pesos, não nos vieses
                    p -= lr * cfg["weight_decay"] * p
                p -= lr * (m[i] / (1 - b1**t)) / (np.sqrt(v[i] / (1 - b2**t)) + eps)
    return model


# --------------------------------------------------------------------------
# Validação cruzada, treino final e submissão
# --------------------------------------------------------------------------
def cross_validate(X, y, activation, cfg, log=print):
    """K-fold estratificado repetido. O pré-processamento é reajustado em cada fold."""
    f1s, accs = [], []
    oof = np.zeros(len(y))
    for rep in range(cfg["cv_repeats"]):
        rng = np.random.default_rng(cfg["seed"] + rep)
        for k, val in enumerate(stratified_folds(y, cfg["cv_folds"], rng)):
            tr = np.setdiff1d(np.arange(len(y)), val)
            pre = Preprocessor(cfg["n_components"], cfg["snv"]).fit(X[tr])
            model = train_model(pre.transform(X[tr]), y[tr], activation, cfg,
                                seed=cfg["seed"] + 1000 * rep + k)
            p = model.predict_proba(pre.transform(X[val]))
            oof[val] += p / cfg["cv_repeats"]
            pred = (p >= 0.5).astype(int)
            f1s.append(macro_f1(y[val], pred))
            accs.append(float(np.mean(pred == y[val])))
            log(f"  rep {rep + 1} fold {k + 1}: macro-F1 {f1s[-1]:.4f}  acc {accs[-1]:.4f}")
    return {
        "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "macro_f1_per_fold": f1s,
    }, oof


def gradient_check(activation, seed=0):
    """Compara o backprop com diferenças finitas. Devolve o maior erro relativo."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(8, 6))
    y = rng.integers(0, 2, 8).astype(float)
    model = MLP(6, [6, 5], activation, rng, dropout=0.0, leaky_slope=0.01)
    logits, cache = model.forward(X)
    grads = model.backward(bce_with_logits(logits, y)[1], cache)
    worst = 0.0
    for p, g in zip(model.params, grads):
        for i in range(min(p.size, 20)):
            old = p.flat[i]
            p.flat[i] = old + 1e-5
            lp = bce_with_logits(model.forward(X)[0], y)[0]
            p.flat[i] = old - 1e-5
            lm = bce_with_logits(model.forward(X)[0], y)[0]
            p.flat[i] = old
            num = (lp - lm) / 2e-5
            worst = max(worst, abs(num - g.flat[i]) / max(1e-8, abs(num) + abs(g.flat[i])))
    return worst


def main(activation):
    ap = argparse.ArgumentParser(description=f"MLP NumPy com ativação {activation}")
    ap.add_argument("--train", default=os.environ.get("TRAIN_CSV", "/data/train_spectra.csv"))
    ap.add_argument("--test", default=os.environ.get("TEST_CSV", "/data/test_spectra.csv"))
    ap.add_argument("--out", default=os.environ.get("OUT_DIR", "/out"))
    ap.add_argument("--skip-cv", action="store_true", help="pula a validação cruzada")
    ap.add_argument("--gradcheck", action="store_true", help="só confere o backprop e sai")
    args = ap.parse_args()

    if args.gradcheck:
        err = gradient_check(activation)
        print(f"[{activation}] gradcheck: maior erro relativo = {err:.2e}")
        raise SystemExit(0 if err < 1e-5 else 1)

    cfg = dict(CONFIG)
    os.makedirs(args.out, exist_ok=True)
    start = time.time()
    print(f"[{activation}] lendo {args.train}")
    ids, X, y = load_csv(args.train)
    print(f"[{activation}] treino: {X.shape[0]} espectros x {X.shape[1]} pontos, "
          f"classes {np.bincount(y).tolist()}")

    metrics = {"activation": activation, "config": cfg}
    if not args.skip_cv:
        print(f"[{activation}] validação cruzada {cfg['cv_repeats']}x{cfg['cv_folds']} (só treino)")
        cv, oof = cross_validate(X, y.astype(float), activation, cfg)
        metrics["cv"] = cv
        print(f"[{activation}] CV macro-F1 = {cv['macro_f1_mean']:.4f} ± {cv['macro_f1_std']:.4f}"
              f" | acc = {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}")
        np.savetxt(os.path.join(args.out, "oof_predictions.csv"),
                   np.column_stack([ids, y, oof]), delimiter=",",
                   header="id,class,prob_diabetes", comments="", fmt=["%d", "%d", "%.6f"])

    print(f"[{activation}] treino final: {cfg['ensemble']} modelos em todo o conjunto de treino")
    pre = Preprocessor(cfg["n_components"], cfg["snv"]).fit(X)
    Xt = pre.transform(X)
    models = [train_model(Xt, y.astype(float), activation, cfg, seed=cfg["seed"] + 7 * i)
              for i in range(cfg["ensemble"])]
    metrics["n_params"] = models[0].n_params()
    train_prob = np.mean([m.predict_proba(Xt) for m in models], axis=0)
    metrics["train_macro_f1"] = macro_f1(y, (train_prob >= 0.5).astype(int))

    if os.path.exists(args.test):
        test_ids, X_test, _ = load_csv(args.test)
        prob = np.mean([m.predict_proba(pre.transform(X_test)) for m in models], axis=0)
        pred = (prob >= 0.5).astype(int)
        sub = os.path.join(args.out, "submission.csv")
        with open(sub, "w") as f:
            f.write("id,class\n")
            for i, c in zip(test_ids, pred):
                f.write(f"{i},{c}\n")
        metrics["test_predicted_counts"] = np.bincount(pred, minlength=2).tolist()
        print(f"[{activation}] submissão: {sub} ({len(pred)} linhas, "
              f"classes previstas {metrics['test_predicted_counts']})")
    else:
        print(f"[{activation}] {args.test} não encontrado: submissão não gerada")

    metrics["seconds"] = round(time.time() - start, 1)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{activation}] pronto em {metrics['seconds']}s")
