"""Compara os experimentos a partir de outputs/<ativacao>/metrics.json."""
import json
import os

rows = []
for act in ("svm", "cnn", "leaky_relu", "gelu", "swiglu"):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", act, "metrics.json")
    if not os.path.exists(path):
        print(f"{act}: ainda não rodou ({path} não existe)")
        continue
    m = json.load(open(path))
    if "cv" in m:
        rows.append((m["cv"]["macro_f1_mean"], m["cv"]["macro_f1_std"], m["cv"]["accuracy_mean"], act, m))

print(f"{'modelo':<12}{'macro-F1 (CV)':<20}{'acurácia (CV)':<16}previsto no teste [0, 1]")
for f1, std, acc, act, m in sorted(rows, reverse=True):
    print(f"{act:<12}{f'{f1:.4f} ± {std:.4f}':<20}{acc:<16.4f}{m.get('test_predicted_counts', '-')}")
if rows:
    print(f"\nMelhor na validação cruzada: {max(rows)[3]} -> outputs/{max(rows)[3]}/submission.csv")
