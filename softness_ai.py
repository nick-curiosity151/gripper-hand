"""softness_ai.py - 記録した触覚データから、AIが柔らかさを見分けられるかを確かめる

使い方:  python softness_ai.py [logsフォルダ]   (numpy だけで動きます)

grasp_runner.py が保存した logs/trials.csv と logs/raw_*.csv を読み、
FSRが最初に反応した(80以上)時点から
  dw : 指がどれだけ進んだか [mm] (サーボの位置フィードバックから)
  Fp : 指先の力の最大値 (FSR)
を取り出して、ロジスティック回帰で sponge(柔らかい) / hayabusa(硬い) を判定します。
評価は一個抜き交差検証 (1回分を抜いて学習し、抜いた1回を当てる) です。
物体の大きさ(カメラの幅や指の絶対位置)は使いません。
"""
import csv, os, sys
import numpy as np

TOUCH = 80
LOG = sys.argv[1] if len(sys.argv) > 1 else "logs"
EXCLUDE_DATE = ("20261122",)   # デモ撮影の日の記録は使わない


def features():
    rows = []
    for t in csv.DictReader(open(os.path.join(LOG, "trials.csv"), encoding="utf-8")):
        f = os.path.join(LOG, t["raw_file"])
        if t["tag"] == "v1" or t["time"].startswith(EXCLUDE_DATE):
            continue
        if not os.path.exists(f) or os.path.getsize(f) < 200:
            continue
        R = list(csv.DictReader(open(f, encoding="utf-8")))
        F = np.array([max(int(r["fsrL"]), int(r["fsrR"])) for r in R], float)
        W = np.array([float(r["wmm"]) for r in R])
        hit = np.where(F >= TOUCH)[0]
        if len(hit) == 0 or hit[0] < 2:
            continue
        i0 = hit[0]
        rows.append((t["condition"], t["object"], W[i0] - W[i0:].min(), F[i0:].max()))
    return rows


def fit(X, y, l2=0.1, it=3000, lr=0.1):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ w + b))); g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w); b -= lr * g.mean()
    return w, b


def loo(rows):
    X = np.array([[r[2], r[3]] for r in rows]); y = np.array([r[1] == "sponge" for r in rows], float)
    ok = 0
    for i in range(len(rows)):
        m = np.arange(len(rows)) != i
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-9
        w, b = fit((X[m] - mu) / sd, y[m])
        ok += ((((X[i] - mu) / sd) @ w + b) > 0) == y[i]
    return int(ok), len(rows)


if __name__ == "__main__":
    data = features()
    for cond, name in [("fixed", "一定の力で押し込んだとき"), ("vision_touch", "触れてすぐ止めたとき")]:
        ok, n = loo([r for r in data if r[0] == cond])
        print(f"{name}: {ok}/{n} 正解 ({ok / n:.0%})")
