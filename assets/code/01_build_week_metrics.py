import numpy as np
import pandas as pd

# =========================
# Config (你可以先不改)
# =========================
DATA_CSV = "/Users/zmpark/Desktop/MCM2026C_code/2026_MCM_Problem_C_Data.csv"
OUT_CSV = "task2_week_metrics.csv"

ASSUME_BOTTOM2_START_SEASON = 28  # 题目建议合理假设 S28 开始 bottom-two + judges choice

# Monte Carlo parameters (先用这个，能跑得动；想更稳可以加大)
M_DRAWS = 3000          # 每个 season-week 采样次数
MIN_ACCEPT = 80         # 接受样本太少就认为该周不稳定/不可用
EPS = 1e-9

# Prior hyperparameters (你们后面可以替换成网格搜索后的值)
BETA = 0.6
RHO = 0.4
KAPPA = 20.0

# =========================
# Utilities
# =========================
def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + EPS)

def entropy(p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0)
    return float(-np.sum(p * np.log(p)))

def judge_total_cols(df: pd.DataFrame) -> list[str]:
    # columns like weekX_judgeY_score
    return [c for c in df.columns if c.startswith("week") and "_judge" in c and c.endswith("_score")]

def parse_week_from_col(col: str) -> int:
    # "week3_judge1_score" -> 3
    return int(col.split("_")[0].replace("week", ""))

def regime_for_season(season: int) -> str:
    if season <= 2:
        return "S1-2 (Rank)"
    if 3 <= season <= 27:
        return "S3-27 (Percent)"
    return "S28-34 (Rank+Bottom2)"

def ranks_average(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    # returns average ranks (1 = best)
    s = pd.Series(values)
    if higher_is_better:
        return s.rank(ascending=False, method="average").to_numpy()
    else:
        return s.rank(ascending=True, method="average").to_numpy()

def elim_percent(J: np.ndarray, v: np.ndarray, k: int) -> set[int]:
    # u = J / sum J ; c = u + v ; eliminate k smallest c
    u = J / (np.sum(J) + EPS)
    c = u + v
    idx = np.argsort(c)  # ascending: worst first
    return set(idx[:k]) if k > 0 else set()

def elim_rank(J: np.ndarray, v: np.ndarray, k: int) -> set[int]:
    rJ = ranks_average(J, higher_is_better=True)     # 1 best
    rV = ranks_average(v, higher_is_better=True)     # 1 best
    R = rJ + rV
    idx = np.argsort(R)[::-1]  # descending: worst first (largest R)
    return set(idx[:k]) if k > 0 else set()

def bottom2_percent(J: np.ndarray, v: np.ndarray) -> list[int]:
    u = J / (np.sum(J) + EPS)
    c = u + v
    idx = np.argsort(c)
    return list(idx[:2])

def bottom2_rank(J: np.ndarray, v: np.ndarray) -> list[int]:
    rJ = ranks_average(J, higher_is_better=True)
    rV = ranks_average(v, higher_is_better=True)
    R = rJ + rV
    idx = np.argsort(R)[::-1]
    return list(idx[:2])

def judges_choice_proxy(bottom2: list[int], J: np.ndarray) -> int:
    # eliminate the one with smaller judges score among bottom two
    a, b = bottom2[0], bottom2[1]
    return a if J[a] < J[b] else b

# =========================
# 1) Load data
# =========================
raw = pd.read_csv(DATA_CSV)

score_cols = judge_total_cols(raw)
if not score_cols:
    raise ValueError("No weekly judge score columns found. Check column names like weekX_judgeY_score.")

# Replace N/A strings etc. with NaN then 0
for c in score_cols:
    raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0.0)

# Identify all weeks present by parsing columns
weeks = sorted({parse_week_from_col(c) for c in score_cols})

# =========================
# 2) Build long table: (season, celebrity) x week -> J_total
# =========================
# Sum judge scores per week across judges
week_to_cols = {w: [c for c in score_cols if parse_week_from_col(c) == w] for w in weeks}

rows = []
for _, r in raw.iterrows():
    season = int(r["season"])
    name = r["celebrity_name"]
    for w in weeks:
        Jw = float(np.sum([r[c] for c in week_to_cols[w]]))
        rows.append((season, name, w, Jw))

dfJ = pd.DataFrame(rows, columns=["season", "celebrity_name", "week", "J"])
# valid week if max J > 0 in that season-week
valid = dfJ.groupby(["season", "week"])["J"].max().reset_index()
valid = valid[valid["J"] > 0].copy()

valid_weeks_by_season = {
    s: sorted(valid.loc[valid["season"] == s, "week"].tolist())
    for s in sorted(dfJ["season"].unique())
}

# =========================
# 3) For each season-week: active set, eliminated set inferred
# =========================
# active if J > 0 that week
def active_names(season: int, week: int) -> list[str]:
    sub = dfJ[(dfJ["season"] == season) & (dfJ["week"] == week)]
    return sub.loc[sub["J"] > 0, "celebrity_name"].tolist()

def J_vector(season: int, week: int, names: list[str]) -> np.ndarray:
    sub = dfJ[(dfJ["season"] == season) & (dfJ["week"] == week)].set_index("celebrity_name")
    return sub.loc[names, "J"].to_numpy(dtype=float)

# =========================
# 4) Main loop: rejection sampling -> entropy & flips
# =========================
records = []

for season in sorted(valid_weeks_by_season.keys()):
    weeks_valid = valid_weeks_by_season[season]
    if len(weeks_valid) < 2:
        continue  # too short

    prev_vhat = None

    for idx_w, w in enumerate(weeks_valid[:-1]):  # exclude last valid week (no next week to infer elimination)
        w_next = weeks_valid[idx_w + 1]

        A = active_names(season, w)
        A_next = set(active_names(season, w_next))

        if len(A) <= 2:
            continue  # too few contestants for meaningful elimination compare

        eliminated_names = [nm for nm in A if nm not in A_next]
        k = len(eliminated_names)  # number eliminated

        if k == 0:
            # no elimination week; still can compute entropy/flip using unconstrained sampling (weak)
            # We'll mark as NaN for flips to avoid misleading signals
            records.append({
                "season": season,
                "week": w,
                "entropy": np.nan,
                "flip_pct_vs_rank": np.nan,
                "flip_pct_vs_pctJC": np.nan,
                "flip_rank_vs_rankJC": np.nan,
                "rule_regime": regime_for_season(season),
                "n_active": len(A),
                "k_elim": 0,
                "n_accept": 0
            })
            continue

        # Build J and standardized z
        J = J_vector(season, w, A)
        J_mean = np.mean(J)
        J_sd = np.std(J)
        z = (J - J_mean) / (J_sd + EPS)

        n = len(A)

        # prev vhat for inertia
        if prev_vhat is None or len(prev_vhat) != n:
            prev_v = np.ones(n) / n
        else:
            prev_v = prev_vhat

        mu = softmax(BETA * z + RHO * np.log(prev_v + EPS))
        alpha = KAPPA * mu

        # map eliminated indices in A
        elim_idx_obs = set([A.index(nm) for nm in eliminated_names])

        # determine which "observed-rule regime" applies for acceptance constraint
        if season <= 2:
            accept_mode = "rank_exact"
        elif 3 <= season <= 27:
            accept_mode = "pct_exact"
        else:
            accept_mode = "bottom2_membership"  # for S28+: only require eliminated in bottom2 (per your modeling)

        accepted = []
        rng = np.random.default_rng(2026 + season * 100 + w)

        # rejection sampling
        for _ in range(M_DRAWS):
            v = rng.dirichlet(alpha)

            if accept_mode == "pct_exact":
                if elim_percent(J, v, k) == elim_idx_obs:
                    accepted.append(v)
            elif accept_mode == "rank_exact":
                if elim_rank(J, v, k) == elim_idx_obs:
                    accepted.append(v)
            else:
                # bottom2 membership under rank by default (consistent with your assumption)
                b2 = bottom2_rank(J, v)
                # if multiple elimination week, relax (rare in later seasons): require at least one eliminated in bottom2
                if len(elim_idx_obs.intersection(set(b2))) >= 1:
                    accepted.append(v)

        n_accept = len(accepted)
        if n_accept < MIN_ACCEPT:
            # Not enough evidence; still write row, but mark NaNs to avoid overclaim
            records.append({
                "season": season,
                "week": w,
                "entropy": np.nan,
                "flip_pct_vs_rank": np.nan,
                "flip_pct_vs_pctJC": np.nan,
                "flip_rank_vs_rankJC": np.nan,
                "rule_regime": regime_for_season(season),
                "n_active": len(A),
                "k_elim": k,
                "n_accept": n_accept
            })
            prev_vhat = None
            continue

        V = np.vstack(accepted)
        vhat = V.mean(axis=0)
        prev_vhat = vhat.copy()

        H = entropy(vhat)

        # Counterfactual flips computed on accepted draws
        flip_pr = 0
        flip_pct_jc = 0
        flip_rank_jc = 0

        for v in V:
            e_pct = elim_percent(J, v, k)
            e_rank = elim_rank(J, v, k)
            if e_pct != e_rank:
                flip_pr += 1

            # percent+JC proxy: determine bottom2 by percent then judges choice
            b2p = bottom2_percent(J, v)
            e_pct_jc = judges_choice_proxy(b2p, J)
            # compare with pure percent elimination (single elimination proxy: compare whether the main eliminated differs)
            # if k>1, we compare against the worst single elimination under percent for stability comparison
            if k >= 1:
                e_pct_single = list(elim_percent(J, v, 1))[0]
                if e_pct_single != e_pct_jc:
                    flip_pct_jc += 1

            # rank+JC proxy
            b2r = bottom2_rank(J, v)
            e_rank_jc = judges_choice_proxy(b2r, J)
            e_rank_single = list(elim_rank(J, v, 1))[0]
            if e_rank_single != e_rank_jc:
                flip_rank_jc += 1

        flip_pr = flip_pr / n_accept
        flip_pct_jc = flip_pct_jc / n_accept
        flip_rank_jc = flip_rank_jc / n_accept

        records.append({
            "season": season,
            "week": w,
            "entropy": H,
            "flip_pct_vs_rank": flip_pr,
            "flip_pct_vs_pctJC": flip_pct_jc,
            "flip_rank_vs_rankJC": flip_rank_jc,
            "rule_regime": regime_for_season(season),
            "n_active": len(A),
            "k_elim": k,
            "n_accept": n_accept
        })

# =========================
# 5) Save
# =========================
out = pd.DataFrame(records)
out = out.sort_values(["season", "week"]).reset_index(drop=True)
out.to_csv(OUT_CSV, index=False)
print(f"[Saved] {OUT_CSV}  rows={len(out)}")
print("Columns:", list(out.columns))
