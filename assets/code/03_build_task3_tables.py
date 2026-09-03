import re
import numpy as np
import pandas as pd

RAW_CSV = "/Users/zmpark/Desktop/MCM2026C_code/2026_MCM_Problem_C_Data.csv"

OUT_WEEKLY = "C_task3_model_weekly.csv"
OUT_SURV   = "C_task3_survival.csv"

EPS = 1e-9

def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def extract_weeks_and_judges(columns):
    """
    Parse columns like week3_judge2_score -> (week=3, judge=2)
    """
    pat = re.compile(r"^week(\d+)_judge(\d+)_score$")
    mapping = []
    for c in columns:
        m = pat.match(c)
        if m:
            mapping.append((c, int(m.group(1)), int(m.group(2))))
    return mapping

def build_weekly_long_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    week_judge_cols = extract_weeks_and_judges(raw.columns)
    if not week_judge_cols:
        raise ValueError("No weekX_judgeY_score columns found in RAW_CSV.")

    # Turn wide -> long for judge scores first
    # We'll melt all judge columns, then aggregate by (season, celeb, week)
    score_cols = [c for c,_,_ in week_judge_cols]

    base_cols = [
        "season","celebrity_name","ballroom_partner",
        "celebrity_industry","celebrity_homestate","celebrity_homecountry/region",
        "celebrity_age_during_season","results","placement"
    ]
    missing = [c for c in base_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"RAW missing required columns: {missing}")

    tmp = raw[base_cols + score_cols].copy()

    # Melt to long at judge level
    long_j = tmp.melt(
        id_vars=base_cols,
        value_vars=score_cols,
        var_name="week_judge",
        value_name="score"
    )
    long_j["score"] = safe_numeric(long_j["score"]).fillna(0.0)

    # parse week from column name
    m = long_j["week_judge"].str.extract(r"^week(\d+)_judge(\d+)_score$")
    long_j["week"] = safe_numeric(m[0]).astype(int)
    long_j["judge_id"] = safe_numeric(m[1]).astype(int)

    # Aggregate to weekly contestant totals
    gcols = base_cols + ["week"]
    weekly = long_j.groupby(gcols, as_index=False).agg(
        judge_total=("score","sum"),
        judge_count=("score", lambda x: int((x>0).sum()))
    )

    # Judge mean (avoid 0/0)
    weekly["judge_mean"] = weekly["judge_total"] / weekly["judge_count"].replace(0, np.nan)
    weekly["judge_mean"] = weekly["judge_mean"].fillna(0.0)

    return weekly

def add_valid_week_active_elim(weekly: pd.DataFrame) -> pd.DataFrame:
    # valid week: at least one positive judge_total in that season-week
    week_max = weekly.groupby(["season","week"])["judge_total"].max().reset_index(name="max_total")
    weekly = weekly.merge(week_max, on=["season","week"], how="left")
    weekly["week_valid"] = weekly["max_total"] > 0
    weekly.drop(columns=["max_total"], inplace=True)

    # active if judge_total>0 (your current definition)
    weekly["active"] = weekly["judge_total"] > 0

    # n_active per season-week (only among valid weeks)
    n_active = weekly[weekly["week_valid"]].groupby(["season","week"])["active"].sum().reset_index(name="n_active")
    weekly = weekly.merge(n_active, on=["season","week"], how="left")
    weekly["n_active"] = weekly["n_active"].fillna(0).astype(int)

    # elimination inference: if active in week w and becomes inactive in next VALID week
    weekly = weekly.sort_values(["season","celebrity_name","week"]).reset_index(drop=True)

    # Build next valid week lookup within each season
    # For each season, list valid weeks in order, map each valid week to its next valid week
    next_valid_map = {}
    for s, g in weekly.groupby("season"):
        valid_weeks = sorted(g.loc[g["week_valid"], "week"].unique().tolist())
        for i, w in enumerate(valid_weeks):
            next_w = valid_weeks[i+1] if i+1 < len(valid_weeks) else None
            next_valid_map[(s, w)] = next_w

    def infer_elim_flag(row, grp):
        # row is at some week; only consider valid weeks
        if (not row["week_valid"]) or (not row["active"]):
            return False
        nxt = next_valid_map.get((row["season"], row["week"]), None)
        if nxt is None:
            return False
        # find active status at next valid week for same contestant
        nxt_row = grp.get(nxt, None)
        if nxt_row is None:
            return False
        return (nxt_row["active"] == False)

    # create a dict per contestant for fast lookup
    elim_flags = []
    for (s, celeb), g in weekly.groupby(["season","celebrity_name"]):
        # map week -> row dict
        by_week = {int(r["week"]): r for _, r in g.iterrows()}
        for _, r in g.iterrows():
            elim_flags.append(infer_elim_flag(r, by_week))
    weekly["elim_this_week"] = elim_flags

    # k_elim per season-week
    k_elim = weekly.groupby(["season","week"])["elim_this_week"].sum().reset_index(name="k_elim")
    weekly = weekly.merge(k_elim, on=["season","week"], how="left")
    weekly["k_elim"] = weekly["k_elim"].fillna(0).astype(int)

    return weekly

def add_judge_z(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Week-within-season standardization among ACTIVE contestants in VALID weeks:
    judge_z = (judge_total - mean_active) / (sd_active + eps)
    Inactive rows keep NaN (so they don't contaminate regressions).
    """
    weekly["judge_z"] = np.nan

    mask = weekly["week_valid"] & weekly["active"]
    grp = weekly[mask].groupby(["season","week"])["judge_total"]
    mean = grp.transform("mean")
    sd = grp.transform("std").replace(0, np.nan)

    z = (weekly.loc[mask, "judge_total"] - mean) / (sd + EPS)
    weekly.loc[mask, "judge_z"] = z

    return weekly

def build_survival_table(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (season, celebrity): last active valid week, event indicator, time-to-event
    """
    base_cols = [
        "season","celebrity_name","ballroom_partner","celebrity_industry",
        "celebrity_homestate","celebrity_homecountry/region","celebrity_age_during_season",
        "results","placement"
    ]
    # Only valid weeks
    wv = weekly[weekly["week_valid"]].copy()

    # last active week
    last_active = wv[wv["active"]].groupby(["season","celebrity_name"])["week"].max().reset_index(name="last_active_week")
    season_last = wv.groupby("season")["week"].max().reset_index(name="season_last_week")

    surv = (wv.groupby(["season","celebrity_name"], as_index=False)[base_cols].first()
            .merge(last_active, on=["season","celebrity_name"], how="left")
            .merge(season_last, on="season", how="left"))

    # event: 1 if eliminated before season end (i.e., last_active_week < season_last_week), else 0
    surv["last_active_week"] = surv["last_active_week"].fillna(0).astype(int)
    surv["event"] = (surv["last_active_week"] < surv["season_last_week"]).astype(int)

    # time: discrete survival time (weeks survived in valid weeks)
    surv["time"] = surv["last_active_week"]

    return surv

def main():
    raw = pd.read_csv(RAW_CSV)

    weekly = build_weekly_long_from_raw(raw)
    weekly = add_valid_week_active_elim(weekly)
    weekly = add_judge_z(weekly)

    # Reserve columns for Task1 fan vote estimates (merge later)
    # v_mean: posterior mean vote share; v_sd: posterior sd; v_logit: logit transform
    weekly["v_mean"] = np.nan
    weekly["v_sd"] = np.nan
    weekly["v_logit"] = np.nan  # to be filled after merging v_mean

    # Save weekly model-ready table
    weekly.to_csv(OUT_WEEKLY, index=False)

    # Build survival table
    surv = build_survival_table(weekly)

    # Reserve summary fan vote columns for survival-level modeling later
    surv["v_avg_mean"] = np.nan
    surv["v_avg_sd"] = np.nan

    surv.to_csv(OUT_SURV, index=False)

    print("[Saved]", OUT_WEEKLY, "shape =", weekly.shape)
    print("[Saved]", OUT_SURV, "shape =", surv.shape)
    print("Next: merge Task1 fan vote results into v_mean/v_sd by (season, week, celebrity_name).")

if __name__ == "__main__":
    main()
