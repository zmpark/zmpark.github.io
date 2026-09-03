import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
INPUT_CSV = "task2_week_metrics.csv"   # 你导出的周级指标表
OUT_DIR = "figures_out"                # 输出目录
DPI = 300

# -----------------------------
# Helpers
# -----------------------------
def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def add_light_grid(ax):
    ax.grid(True, which="major", linewidth=0.6, alpha=0.25)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.15)
    ax.minorticks_on()

def annotate_topk(ax, x, y, labels, k=5, xytext=(6, 6)):
    """标注y最大/最关键的k个点（避免图太空）"""
    if len(y) == 0:
        return
    idx = np.argsort(y)[-k:]
    for i in idx:
        ax.annotate(
            str(labels[i]),
            (x[i], y[i]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=9,
            alpha=0.9
        )

# -----------------------------
# Load & validate
# -----------------------------
df = pd.read_csv(INPUT_CSV)

required_cols = {"season", "week", "entropy", "flip_pct_vs_rank"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {INPUT_CSV}: {missing}")

# Ensure numeric
for c in ["season", "week", "entropy", "flip_pct_vs_rank"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["season", "week", "entropy", "flip_pct_vs_rank"]).copy()

# Optional columns
has_pct_jc  = "flip_pct_vs_pctJC" in df.columns
has_rank_jc = "flip_rank_vs_rankJC" in df.columns
has_regime  = "rule_regime" in df.columns

ensure_out_dir(OUT_DIR)

# ============================================================
# FIGURE A: Season-level rule disagreement (EFR)
# ============================================================
# EFR(s) = average flip probability across weeks in season s
season_grp = df.groupby("season", as_index=False).agg(
    efr=("flip_pct_vs_rank", "mean"),
    n_weeks=("flip_pct_vs_rank", "count")
)

# sort by season number (x-axis clean)
season_grp = season_grp.sort_values("season").reset_index(drop=True)

# Add a simple uncertainty band: standard error (optional)
season_std = df.groupby("season")["flip_pct_vs_rank"].std()
season_cnt = df.groupby("season")["flip_pct_vs_rank"].count()
season_se  = (season_std / np.sqrt(season_cnt)).reindex(season_grp["season"]).values
season_se  = np.nan_to_num(season_se, nan=0.0)

fig, ax = plt.subplots(figsize=(11.5, 5.2))
x = season_grp["season"].values
y = season_grp["efr"].values

# Line + markers (default colors)
ax.plot(x, y, marker="o", linewidth=1.6, markersize=4)

# Error bars (very light)
ax.errorbar(x, y, yerr=season_se, fmt="none", capsize=2, alpha=0.35)

ax.set_title("Season-level rule disagreement (EFR): Percent vs Rank", fontsize=13, pad=10)
ax.set_xlabel("Season", fontsize=11)
ax.set_ylabel("EFR(s) = mean flip probability across weeks", fontsize=11)
ax.set_ylim(0, 1)

add_light_grid(ax)

# Annotate top-5 seasons with highest EFR
annotate_topk(ax, x, y, labels=x, k=5)

# Add small note for interpretability
ax.text(
    0.01, 0.02,
    "Higher EFR means outcomes are more sensitive to the choice of voting rule.",
    transform=ax.transAxes,
    fontsize=9,
    alpha=0.8
)

fig.tight_layout()
out_a = os.path.join(OUT_DIR, "figA_EFR_by_season.png")
fig.savefig(out_a, dpi=DPI)
plt.close(fig)

print(f"[Saved] {out_a}")

# ============================================================
# FIGURE B: Flip probability vs Entropy (+ Judges' choice)
# ============================================================
# We'll plot baseline (pct vs rank) as scatter, and optionally overlay JC effects.

# For visual clarity, we can lightly bin entropy and show trend (moving average)
def moving_average(xv, yv, window=25):
    """simple moving average after sorting by x"""
    order = np.argsort(xv)
    xs = xv[order]
    ys = yv[order]
    if len(xs) < window:
        return xs, ys
    ma = np.convolve(ys, np.ones(window)/window, mode="valid")
    xm = xs[window-1:]
    return xm, ma

fig, ax = plt.subplots(figsize=(7.6, 5.6))

# Baseline scatter
ax.scatter(
    df["entropy"].values,
    df["flip_pct_vs_rank"].values,
    s=18,
    alpha=0.5,
    marker="o",
    label="Percent vs Rank (baseline)"
)

# Trend line (baseline)
xm, ym = moving_average(df["entropy"].values, df["flip_pct_vs_rank"].values, window=35)
ax.plot(xm, ym, linewidth=2.0, label="Baseline trend (moving avg)")

# If JC columns exist, overlay their flip probabilities (as separate scatter + trend)
if has_pct_jc:
    df["flip_pct_vs_pctJC"] = pd.to_numeric(df["flip_pct_vs_pctJC"], errors="coerce")
    d2 = df.dropna(subset=["flip_pct_vs_pctJC"]).copy()
    ax.scatter(
        d2["entropy"].values,
        d2["flip_pct_vs_pctJC"].values,
        s=18,
        alpha=0.5,
        marker="^",
        label="Percent vs Percent+JC"
    )
    xm2, ym2 = moving_average(d2["entropy"].values, d2["flip_pct_vs_pctJC"].values, window=35)
    ax.plot(xm2, ym2, linewidth=2.0, label="Percent+JC trend (moving avg)")

if has_rank_jc:
    df["flip_rank_vs_rankJC"] = pd.to_numeric(df["flip_rank_vs_rankJC"], errors="coerce")
    d3 = df.dropna(subset=["flip_rank_vs_rankJC"]).copy()
    ax.scatter(
        d3["entropy"].values,
        d3["flip_rank_vs_rankJC"].values,
        s=18,
        alpha=0.5,
        marker="s",
        label="Rank vs Rank+JC"
    )
    xm3, ym3 = moving_average(d3["entropy"].values, d3["flip_rank_vs_rankJC"].values, window=35)
    ax.plot(xm3, ym3, linewidth=2.0, label="Rank+JC trend (moving avg)")

ax.set_title("When does judges' choice stabilize outcomes?", fontsize=13, pad=10)
ax.set_xlabel("Posterior entropy  H( v̂ )  (higher = more diffuse fan support)", fontsize=11)
ax.set_ylabel("Flip probability", fontsize=11)
ax.set_ylim(0, 1)

add_light_grid(ax)
ax.legend(frameon=False, fontsize=9, loc="upper left")

# Highlight top-right region (high entropy & high flip): where system is most unstable
ax.text(
    0.62, 0.08,
    "High-entropy weeks\nare typically\nmore rule-sensitive",
    transform=ax.transAxes,
    fontsize=9,
    alpha=0.85
)

fig.tight_layout()
out_b = os.path.join(OUT_DIR, "figB_flip_vs_entropy_JC.png")
fig.savefig(out_b, dpi=DPI)
plt.close(fig)

print(f"[Saved] {out_b}")

print("\nDone. Two figures generated in:", OUT_DIR)
