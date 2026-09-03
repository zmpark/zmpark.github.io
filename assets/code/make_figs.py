import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILENAME = "2026_MCM_Problem_C_Data.csv"
CSV_PATH = os.path.join(SCRIPT_DIR, CSV_FILENAME)

OUT_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

M_ECS = 2000
M_POST = 30000
EPS = 1e-9

BETA = 0.8
RHO = 0.8
KAPPA = 30.0

SEASON_FOR_FIG2 = 27


def fail_with_help(msg: str):
    print("\n[ERROR]", msg)
    print("\n[Debug info]")
    print("  Script directory:", SCRIPT_DIR)
    print("  Current working directory:", os.getcwd())
    print("\n[Files in script directory]")
    try:
        for f in sorted(os.listdir(SCRIPT_DIR))[:300]:
            print("  -", f)
    except Exception as e:
        print("  (Could not list files:", e, ")")
    print("\n[How to fix]")
    print(f"  Put '{CSV_FILENAME}' into:")
    print(f"  {SCRIPT_DIR}")
    raise SystemExit(1)


def find_week_judge_cols(df):
    pat = re.compile(r"week(\d+)_judge(\d+)_score")
    week_cols = {}
    for c in df.columns:
        m = pat.fullmatch(c)
        if m:
            w = int(m.group(1))
            week_cols.setdefault(w, []).append(c)
    for w in week_cols:
        week_cols[w] = sorted(week_cols[w], key=lambda x: int(re.findall(r"judge(\d+)", x)[0]))
    return dict(sorted(week_cols.items(), key=lambda kv: kv[0]))


def average_rank_desc(values):
    s = pd.Series(values)
    return s.rank(ascending=False, method="average").to_numpy()


def compute_mu(J, v_prev, beta=BETA, rho=RHO, eps=EPS):
    J = np.asarray(J, dtype=float)
    v_prev = np.asarray(v_prev, dtype=float)
    z = (J - J.mean()) / (J.std(ddof=0) + eps)
    logit = beta * z + rho * np.log(v_prev + eps)
    logit = logit - logit.max()
    mu = np.exp(logit)
    mu = mu / (mu.sum() + eps)
    return mu


def alpha_from_mu(mu, kappa=KAPPA, eps=EPS):
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, eps, 1.0)
    mu = mu / mu.sum()
    return kappa * mu


def season_scheme(season):
    # (scheme, judge_save_flag)
    if 3 <= season <= 27:
        return "percent", False
    if season in (1, 2):
        return "rank", False
    return "rank", True  # assumed S28-34 bottom2+judges


def regime_label(season):
    if season in (1, 2):
        return "Rank: S1–2"
    if 3 <= season <= 27:
        return "Percent: S3–27"
    return "Bottom2+Judges: S28–34"


def percent_rule_elim(J, V, k):
    if k == 0:
        return set()
    u = J / (J.sum() + EPS)
    c = u + V
    return set(np.argsort(c)[:k].tolist())


def rank_rule_elim(J, V, k):
    if k == 0:
        return set()
    rJ = average_rank_desc(J)      # 1 best
    rV = average_rank_desc(V)      # 1 best (largest vote share)
    R = rJ + rV
    return set(np.argsort(R)[-k:].tolist())  # worst k


def bottom2_percent(J, V):
    u = J / (J.sum() + EPS)
    c = u + V
    return set(np.argsort(c)[:2].tolist())


def bottom2_rank(J, V):
    rJ = average_rank_desc(J)
    rV = average_rank_desc(V)
    R = rJ + rV
    return set(np.argsort(R)[-2:].tolist())


def rejection_sampling_week(J, elim_set_local, scheme, judge_save, mu, M):
    alpha = alpha_from_mu(mu)
    accepted = []
    for _ in range(M):
        v = np.random.dirichlet(alpha)
        k = len(elim_set_local)

        if scheme == "percent":
            if judge_save and k == 1:
                ok = list(elim_set_local)[0] in bottom2_percent(J, v)
            else:
                ok = (percent_rule_elim(J, v, k) == elim_set_local)
        else:
            if judge_save and k == 1:
                ok = list(elim_set_local)[0] in bottom2_rank(J, v)
            else:
                ok = (rank_rule_elim(J, v, k) == elim_set_local)

        if ok:
            accepted.append(v)

    if len(accepted) == 0:
        return 0, None, None

    accepted = np.asarray(accepted)
    v_mean = accepted.mean(axis=0)
    H = -np.sum(v_mean * np.log(v_mean + EPS))
    return len(accepted), accepted, H


def friendly_name(name: str) -> str:
    """
    拟人短名：不出现省略号。
    - "Juan Pablo Di Pace" -> "Juan P."
    - "Mary Lou Retton" -> "Mary L."
    """
    if not isinstance(name, str):
        name = str(name)
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    if len(parts) == 0:
        return "Unknown"
    if len(parts) == 1:
        return parts[0]
    first = parts[0]
    last_initial = parts[-1][0].upper()
    return f"{first} {last_initial}."



if not os.path.exists(CSV_PATH):
    fail_with_help(f"CSV not found at: {CSV_PATH}")

print("[OK] Found CSV:", CSV_PATH)
df_raw = pd.read_csv(CSV_PATH)

week_cols = find_week_judge_cols(df_raw)
if not week_cols:
    fail_with_help("Could not find columns like weekX_judgeY_score in your CSV.")

for w, cols in week_cols.items():
    for c in cols:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce").fillna(0.0)

if "season" not in df_raw.columns or "celebrity_name" not in df_raw.columns:
    fail_with_help("CSV missing required columns: season and/or celebrity_name.")

seasons = sorted(df_raw["season"].dropna().unique().astype(int).tolist())

np.random.seed(0)
ecs_records = []

for s in seasons:
    df_s = df_raw[df_raw["season"] == s].copy()

    Jmat = {}
    for w, cols in week_cols.items():
        Jmat[w] = df_s[cols].sum(axis=1).to_numpy(dtype=float)

    valid_weeks = [w for w in sorted(Jmat.keys()) if np.max(Jmat[w]) > 0]
    if len(valid_weeks) < 2:
        continue

    prev_v_mean = None

    for t in range(len(valid_weeks) - 1):
        w = valid_weeks[t]
        w_next = valid_weeks[t + 1]

        Jw = Jmat[w]
        Jnext = Jmat[w_next]

        active = Jw > 0
        active_next = Jnext > 0

        active_idx = np.where(active)[0].tolist()
        active_next_idx = set(np.where(active_next)[0].tolist())

        elim_global = set([i for i in active_idx if i not in active_next_idx])
        k = len(elim_global)

        J_active = Jw[active]
        n = len(J_active)
        if n == 0:
            continue

        g2l = {g: j for j, g in enumerate(active_idx)}
        elim_local = set(g2l[g] for g in elim_global)

        v_prev = np.ones(n) / n if prev_v_mean is None else np.ones(n) / n
        mu = compute_mu(J_active, v_prev)

        scheme, judge_save = season_scheme(s)
        Nacc, acc_samples, H = rejection_sampling_week(J_active, elim_local, scheme, judge_save, mu, M_ECS)
        ECS = Nacc / M_ECS

        # ---- FIX: u_elim added ----
        u_elim = np.nan
        if k == 1:
            e = list(elim_local)[0]
            u = J_active / (J_active.sum() + EPS)
            u_elim = float(u[e])

        ecs_records.append({
            "season": s,
            "week": w,
            "regime": regime_label(s),
            "ECS": ECS,
            "H": H if H is not None else np.nan,
            "u_elim": u_elim,
            "k_elim": k
        })

        if acc_samples is not None:
            prev_v_mean = acc_samples.mean(axis=0)

ecs_df = pd.DataFrame(ecs_records).dropna(subset=["ECS"]).reset_index(drop=True)
if len(ecs_df) == 0:
    fail_with_help("ECS table is empty.")

ecs = ecs_df["ECS"].to_numpy(dtype=float)
q10, q25, q50 = np.quantile(ecs, [0.10, 0.25, 0.50])

fig = plt.figure(figsize=(11.5, 7.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=0.35)

# --- Top: histogram ---
ax1 = fig.add_subplot(gs[0, 0])
bins = np.linspace(0, 1, 21)
counts, edges = np.histogram(ecs, bins=bins)
centers = 0.5 * (edges[:-1] + edges[1:])

cmap = plt.cm.viridis
norm = (centers - centers.min()) / (centers.max() - centers.min() + EPS)
colors = cmap(norm)

ax1.bar(centers, counts, width=(edges[1] - edges[0]) * 0.92, color=colors, edgecolor="white", linewidth=0.8)
ax1.axvline(q10, color="#d62728", linewidth=2, linestyle="--", label="10% quantile")
ax1.axvline(q25, color="#ff7f0e", linewidth=2, linestyle="--", label="25% quantile")
ax1.axvline(q50, color="#2ca02c", linewidth=2, linestyle="-", label="median")

ax1.set_title("Distribution of Elimination Consistency Score (ECS)", fontsize=16, pad=10)
ax1.set_xlabel("ECS (higher = more rule-consistent evidence)", fontsize=12)
ax1.set_ylabel("Count of season-weeks", fontsize=12)
ax1.legend(frameon=False, ncol=3, fontsize=10)

# --- Bottom: regime violin + jitter ---
ax2 = fig.add_subplot(gs[1, 0])
regimes = ["Rank: S1–2", "Percent: S3–27", "Bottom2+Judges: S28–34"]
data = [ecs_df.loc[ecs_df["regime"] == r, "ECS"].to_numpy(dtype=float) for r in regimes]

vp = ax2.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
for body, col in zip(vp["bodies"], ["#1f77b4", "#9467bd", "#8c564b"]):
    body.set_facecolor(col)
    body.set_edgecolor("black")
    body.set_alpha(0.35)

vp["cmedians"].set_color("black")
vp["cmedians"].set_linewidth(2)

rng = np.random.default_rng(0)
for i, arr in enumerate(data, start=1):
    if len(arr) == 0:
        continue
    xj = i + rng.normal(0, 0.06, size=len(arr))
    ax2.scatter(xj, arr, s=12, alpha=0.45, color="black")

ax2.set_xticks(range(1, len(regimes) + 1))
ax2.set_xticklabels(regimes, fontsize=11)
ax2.set_ylabel("ECS", fontsize=12)
ax2.set_title("ECS by rule era (mechanism regime)", fontsize=14, pad=8)
ax2.set_ylim(-0.02, 1.02)

fig.savefig(os.path.join(OUT_DIR, "fig1_ecs.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# histogram only (optional)
fig_h = plt.figure(figsize=(10, 4.8))
axh = fig_h.add_subplot(111)
axh.bar(centers, counts, width=(edges[1] - edges[0]) * 0.92, color=colors, edgecolor="white", linewidth=0.8)
axh.axvline(q10, color="#d62728", linewidth=2, linestyle="--", label="10% quantile")
axh.axvline(q25, color="#ff7f0e", linewidth=2, linestyle="--", label="25% quantile")
axh.axvline(q50, color="#2ca02c", linewidth=2, linestyle="-", label="median")
axh.set_title("Distribution of ECS", fontsize=15, pad=10)
axh.set_xlabel("ECS", fontsize=12)
axh.set_ylabel("Count", fontsize=12)
axh.legend(frameon=False, ncol=3, fontsize=10)
fig_h.savefig(os.path.join(OUT_DIR, "fig1_ecs_hist.png"), dpi=300, bbox_inches="tight")
plt.close(fig_h)


# FIG 2: Posterior vote-share distributions (clean names)
df_s = ecs_df[ecs_df["season"] == SEASON_FOR_FIG2].copy().sort_values("ECS")
if len(df_s) == 0:
    SEASON_FOR_FIG2 = int(ecs_df.sort_values("ECS").iloc[0]["season"])
    df_s = ecs_df[ecs_df["season"] == SEASON_FOR_FIG2].copy().sort_values("ECS")

chosen_week = int(df_s.iloc[0]["week"])

df_s_full = df_raw[df_raw["season"] == SEASON_FOR_FIG2].copy()
for ww, cols in week_cols.items():
    df_s_full[f"J_week{ww}"] = df_s_full[cols].sum(axis=1).to_numpy(dtype=float)

valid_weeks = [ww for ww in sorted(week_cols.keys()) if df_s_full[f"J_week{ww}"].max() > 0]
if chosen_week not in valid_weeks:
    chosen_week = valid_weeks[0]
idx = valid_weeks.index(chosen_week)
if idx == len(valid_weeks) - 1:
    idx -= 1
w = valid_weeks[idx]
w_next = valid_weeks[idx + 1]

Jw_all = df_s_full[f"J_week{w}"].to_numpy(dtype=float)
Jnext_all = df_s_full[f"J_week{w_next}"].to_numpy(dtype=float)

active = Jw_all > 0
active_next = Jnext_all > 0
active_idx = np.where(active)[0].tolist()
active_next_idx = set(np.where(active_next)[0].tolist())
elim_global = set([i for i in active_idx if i not in active_next_idx])

J_active = Jw_all[active]
n = len(J_active)

g2l = {g: j for j, g in enumerate(active_idx)}
elim_local = set(g2l[g] for g in elim_global)

mu = compute_mu(J_active, np.ones(n) / n)
scheme, judge_save = season_scheme(SEASON_FOR_FIG2)
Nacc, acc_samples, _ = rejection_sampling_week(J_active, elim_local, scheme, judge_save, mu, M_POST)
if acc_samples is None:
    acc_samples = np.random.dirichlet(alpha_from_mu(mu), size=2000)

v_mean = acc_samples.mean(axis=0)

names = df_s_full.loc[active, "celebrity_name"].tolist()
nice_names = [friendly_name(nm) for nm in names]
data = [acc_samples[:, i] for i in range(n)]

fig2 = plt.figure(figsize=(11.5, 5.4))
ax = fig2.add_subplot(111)

parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
for body in parts["bodies"]:
    body.set_facecolor("#4C78A8")
    body.set_edgecolor("black")
    body.set_alpha(0.25)

bp = ax.boxplot(data, widths=0.18, vert=True, showfliers=False, patch_artist=True)
for box in bp["boxes"]:
    box.set_facecolor("#F58518")
    box.set_alpha(0.25)
for med in bp["medians"]:
    med.set_color("black")
    med.set_linewidth(1.8)

for e in elim_local:
    ax.scatter([e + 1], [v_mean[e]], marker="*", s=180, color="#E45756",
               edgecolor="black", linewidth=0.5, zorder=5)

ax.set_xticks(np.arange(1, n + 1))
ax.set_xticklabels(nice_names, rotation=25, ha="right", fontsize=11)
ax.set_ylabel("Posterior fan vote share", fontsize=12)
ax.set_title(f"Posterior vote-share distributions (Season {SEASON_FOR_FIG2}, Week {w})",
             fontsize=15, pad=10)
ax.text(0.99, 0.98, "★ = eliminated contestant", transform=ax.transAxes,
        ha="right", va="top", fontsize=11)

fig2.savefig(os.path.join(OUT_DIR, "fig2_posterior_violin.png"), dpi=300, bbox_inches="tight")
plt.close(fig2)

# FIG 4: Bridge (3D + projections), colored by entropy
# Layout-improved version: dedicate a column for colorbar -> 3D shifts left & looks balanced
# Fix: move colorbar label/ticks to LEFT to avoid overlapping projections

bridge = ecs_df.dropna(subset=["u_elim", "H"]).copy()
bridge = bridge[np.isfinite(bridge["H"])].copy()

if len(bridge) < 5:
    print("[WARN] Not enough bridge points for Figure 4. Skipping fig4.")
else:
    x = bridge["ECS"].to_numpy(dtype=float)
    y = bridge["u_elim"].to_numpy(dtype=float)
    z = bridge["H"].to_numpy(dtype=float)

    cmap = plt.cm.viridis

    # bigger canvas + more spacing
    fig4 = plt.figure(figsize=(14.8, 7.0))

    # Layout: [3D | colorbar | projections]
    gs4 = gridspec.GridSpec(
        2, 3,
        width_ratios=[3.30, 0.24, 1.35],  # colorbar column wide enough
        height_ratios=[1.0, 1.0],
        wspace=0.48,  # IMPORTANT: add horizontal breathing room
        hspace=0.36
    )

    # --- 3D scatter (left, spans both rows) ---
    ax3d = fig4.add_subplot(gs4[:, 0], projection="3d")
    sc = ax3d.scatter(
        x, y, z,
        s=20,
        c=z, cmap=cmap,
        depthshade=True,
        edgecolor="none",
        alpha=0.95
    )

    ax3d.set_xlabel("ECS (evidence)", labelpad=10)
    ax3d.set_ylabel("Eliminated judges' share $u_{e,w}$", labelpad=10)
    ax3d.set_zlabel("Entropy $H(\\hat{v})$", labelpad=10)
    ax3d.set_title("Bridge: Evidence vs Judges vs Uncertainty (week-level)", pad=12, fontsize=14)
    ax3d.view_init(elev=22, azim=-55)

    # --- Colorbar axis (middle, spans both rows) ---
    cax = fig4.add_subplot(gs4[:, 1])
    cbar = fig4.colorbar(sc, cax=cax)

    # Put label/ticks on LEFT side of colorbar to avoid colliding with projection y-labels
    cbar.set_label("Entropy (higher = more uncertain)", rotation=90, labelpad=18)
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.yaxis.tick_left()
    cbar.ax.tick_params(pad=6)

    # --- Projection: u_e,w vs H (top-right) ---
    ax_yz = fig4.add_subplot(gs4[0, 2])
    ax_yz.scatter(y, z, s=28, c=z, cmap=cmap, edgecolor="none", alpha=0.95)
    ax_yz.set_xlabel("Eliminated judges' share $u_{e,w}$")
    ax_yz.set_ylabel("Entropy $H$")
    ax_yz.set_title("Projection: $u_{e,w}$ vs $H$", fontsize=12, pad=6)

    # --- Projection: ECS vs H (bottom-right) ---
    ax_xz = fig4.add_subplot(gs4[1, 2])
    ax_xz.scatter(x, z, s=28, c=z, cmap=cmap, edgecolor="none", alpha=0.95)
    ax_xz.set_xlabel("ECS")
    ax_xz.set_ylabel("Entropy $H$")
    ax_xz.set_title("Projection: ECS vs $H$", fontsize=12, pad=6)

    # Footnote
    fig4.text(
        0.02, 0.02,
        "Color encodes entropy (posterior uncertainty). Projections reduce 3D perspective ambiguity.",
        fontsize=11
    )

    # margins
    fig4.subplots_adjust(left=0.04, right=0.985, top=0.93, bottom=0.10)

    fig4.savefig(os.path.join(OUT_DIR, "fig4_bridge_3d.png"), dpi=300)
    plt.close(fig4)

    print("  - fig4_bridge_3d.png")
