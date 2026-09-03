import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
INPUT_CSV = "task2_week_metrics.csv"
OUT_DIR = "figures_out"
DPI = 400

N_BINS = 10
MIN_ACCEPT_FILTER = None         # e.g., 80 if you want stricter filtering
KEEP_ONLY_ELIM_WEEKS = True      # keep only k_elim>=1

# =========================
# Helpers
# =========================
def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def add_light_grid(ax):
    ax.grid(True, which="major", linewidth=0.6, alpha=0.18)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.10)
    ax.minorticks_on()

def mean_ci(y: np.ndarray):
    y = y[~np.isnan(y)]
    n = len(y)
    if n <= 1:
        return np.nan, np.nan
    m = float(np.mean(y))
    s = float(np.std(y, ddof=1))
    se = s / np.sqrt(n)
    ci = 1.96 * se
    return m, ci

def binned_stats_quantile(x, y, bins=10):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) == 0:
        return pd.DataFrame(columns=["x_mid", "y_mean", "y_ci", "n"])
    try:
        df["bin"] = pd.qcut(df["x"], bins, duplicates="drop")
    except ValueError:
        edges = np.linspace(df["x"].min(), df["x"].max(), bins + 1)
        df["bin"] = pd.cut(df["x"], edges, include_lowest=True)

    out = []
    for _, g in df.groupby("bin"):
        xv = g["x"].to_numpy()
        yv = g["y"].to_numpy()
        m, ci = mean_ci(yv)
        if np.isnan(m):
            continue
        out.append({"x_mid": float(np.mean(xv)), "y_mean": m, "y_ci": ci, "n": len(g)})
    return pd.DataFrame(out).sort_values("x_mid").reset_index(drop=True)

def text_box(ax, x, y, s, ha="center", va="top", fontsize=12, alpha=0.92):
    """Place text with a subtle white box so it is never blocked by bars."""
    ax.text(
        x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize, alpha=0.9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=alpha)
    )

# =========================
# Load & clean
# =========================
ensure_out_dir(OUT_DIR)

df = pd.read_csv(INPUT_CSV)

for c in ["season", "week", "entropy", "flip_pct_vs_rank", "flip_pct_vs_pctJC",
          "flip_rank_vs_rankJC", "n_accept", "k_elim"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if MIN_ACCEPT_FILTER is not None and "n_accept" in df.columns:
    df = df[df["n_accept"] >= MIN_ACCEPT_FILTER].copy()

if KEEP_ONLY_ELIM_WEEKS and "k_elim" in df.columns:
    df = df[df["k_elim"] >= 1].copy()

# Paper-ish defaults
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

# ============================================================
# Figure A: EFR by season (clean & readable)
# ============================================================
dfA = df.dropna(subset=["flip_pct_vs_rank", "season"]).copy()

season_grp = dfA.groupby("season").agg(
    efr=("flip_pct_vs_rank", "mean"),
    std=("flip_pct_vs_rank", "std"),
    n=("flip_pct_vs_rank", "count")
).reset_index().sort_values("season")

season_grp["ci"] = 1.96 * (season_grp["std"] / np.sqrt(season_grp["n"].clip(lower=1)))
season_grp["ci"] = season_grp["ci"].fillna(0.0)

x = season_grp["season"].to_numpy()
y = season_grp["efr"].to_numpy()
ci = season_grp["ci"].to_numpy()

fig, ax = plt.subplots(figsize=(12.8, 5.6), constrained_layout=True)

# Bars a bit lighter; edges off for a cleaner look
ax.bar(x, y, width=0.88, alpha=0.75, linewidth=0)

# Error bars subtle
ax.errorbar(x, y, yerr=ci, fmt="none", capsize=2, alpha=0.35)

# Regime separators: very light so they don't dominate
ax.axvline(2.5, linewidth=1.0, alpha=0.15)
ax.axvline(27.5, linewidth=1.0, alpha=0.15)

# Regime labels with white boxes so they won't be blocked
text_box(ax, 0.17, 0.96, "S1–2 (Rank)", fontsize=12)
text_box(ax, 0.50, 0.96, "S3–27 (Percent)", fontsize=12)
text_box(ax, 0.85, 0.96, "S28+ (Rank + Bottom-Two)", fontsize=12)

ax.set_title("Season-level rule sensitivity (EFR): Percent vs Rank")
ax.set_xlabel("Season")
ax.set_ylabel("EFR(s) = mean flip probability across weeks")
ax.set_ylim(0, 1)
add_light_grid(ax)

# annotate top seasons with highest EFR (small offset)
topk = 4
idx = np.argsort(y)[-topk:]
for i in idx:
    ax.annotate(
        str(int(x[i])),
        (x[i], y[i]),
        textcoords="offset points",
        xytext=(0, 7),
        ha="center",
        fontsize=10,
        alpha=0.95
    )

# Put the interpretation as a neat footnote-style line
ax.text(
    0.01, -0.12,
    "Note: Higher EFR indicates a higher probability that outcomes change when switching Percent vs Rank.",
    transform=ax.transAxes,
    fontsize=10,
    alpha=0.85
)

out_a = os.path.join(OUT_DIR, "figA_EFR_by_season_FINAL.png")
fig.savefig(out_a, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"[Saved] {out_a}")

# ============================================================
# Figure B: 2-panel summary (baseline + JC stabilization)
# ============================================================
dfB = df.dropna(subset=["entropy", "flip_pct_vs_rank"]).copy()

base_stats = binned_stats_quantile(
    dfB["entropy"].to_numpy(),
    dfB["flip_pct_vs_rank"].to_numpy(),
    bins=N_BINS
)

has_pct_jc = "flip_pct_vs_pctJC" in dfB.columns and dfB["flip_pct_vs_pctJC"].notna().any()
pct_delta_stats = None
if has_pct_jc:
    tmp = dfB.dropna(subset=["flip_pct_vs_pctJC"]).copy()
    tmp["delta_pct"] = tmp["flip_pct_vs_rank"] - tmp["flip_pct_vs_pctJC"]
    pct_delta_stats = binned_stats_quantile(tmp["entropy"].to_numpy(), tmp["delta_pct"].to_numpy(), bins=N_BINS)

fig = plt.figure(figsize=(13.2, 5.6), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# Left: baseline scatter very light + binned curve strong
ax1.scatter(dfB["entropy"], dfB["flip_pct_vs_rank"], s=18, alpha=0.16)
ax1.errorbar(
    base_stats["x_mid"], base_stats["y_mean"], yerr=base_stats["y_ci"],
    fmt="-o", capsize=2, alpha=0.95
)
ax1.set_title("Baseline: rule sensitivity vs uncertainty")
ax1.set_xlabel("Posterior entropy  H( v̂ )")
ax1.set_ylabel("Flip probability (Percent vs Rank)")
ax1.set_ylim(0, 1)
add_light_grid(ax1)
ax1.text(
    0.02, 0.03,
    "Binned mean ± 95% CI (points are season-weeks).",
    transform=ax1.transAxes,
    fontsize=10,
    alpha=0.85
)

# Right: JC delta (clean)
ax2.axhline(0.0, linewidth=1.0, alpha=0.20)
ax2.set_title("Judges' choice stabilization")
ax2.set_xlabel("Posterior entropy  H( v̂ )")
ax2.set_ylabel("Δ flip = baseline − (Percent+JC)")
add_light_grid(ax2)

if pct_delta_stats is not None and len(pct_delta_stats) > 0:
    ax2.errorbar(
        pct_delta_stats["x_mid"], pct_delta_stats["y_mean"], yerr=pct_delta_stats["y_ci"],
        fmt="-o", capsize=2, alpha=0.95,
        label="Δ vs Percent+JC"
    )
    ax2.legend(frameon=False, fontsize=11, loc="upper left")
else:
    ax2.text(
        0.06, 0.55,
        "No Percent+JC column available\n(or too many NaNs after filtering).",
        transform=ax2.transAxes,
        fontsize=11,
        alpha=0.8
    )

ax2.set_ylim(-0.25, 0.35)

fig.suptitle("Uncertainty and rule sensitivity (Task 2 summary)", fontsize=16)

out_b = os.path.join(OUT_DIR, "figB_entropy_flip_FINAL.png")
fig.savefig(out_b, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"[Saved] {out_b}")

print("\nDone. Final figures saved in:", OUT_DIR)
