"""
ingest.py

Combines the three scrapers' saved output into per-gameweek player tables. It only READS
what the scrapers have already written - it never scrapes.

INPUTS (written by the scrapers):
    data/processed/fpl_players.parquet           fpl.py          names, clubs (for matching)
    data/processed/fpl_player_history.parquet    fpl_history.py  one row per player per fixture
    data/raw/fotmob/*/mw*/player_stats.parquet   fotmob_scraper.py  per-match player stats

OUTPUTS (data/processed/):
    player_fixture.parquet   one row per player per fixture (the base grain)
    player_gw.parquet        one row per player per gameweek (double gameweeks summed)
    id_map.csv               FotMob player -> FPL player id. Re-used on every run.

HOW THE JOIN WORKS
  FPL is the base table; FotMob columns are added with an "fm_" prefix so you can always
  see which source a number came from. FotMob rows are linked to FPL rows by
  (FPL player id, opponent, home/away). In a league season a player meets each opponent
  once at home and once away, so that key is unique and does not depend on gameweek
  numbers lining up (which breaks when fixtures are rearranged).

  Players are matched by name in steps, stopping at the first that gives exactly one
  candidate: (1) full or web name equal within the same club, (2) full name equal anywhere
  in the league (covers transfers), then within the club: (3) FPL web name inside the FotMob
  name ("Salah" in "Mohamed Salah"; "B.Fernandes" ~ "Bruno Fernandes"), (4) the FotMob name
  inside FPL's long full name, (5) a close spelling match. Anything left
  is written to id_map.csv with a blank fpl_id. Fill those in by hand and re-run: rows
  that already have an fpl_id are never overwritten.

COLUMN RULES for player_gw.parquet: counting stats are summed (missing FotMob data stays
missing, not zero); price, ownership and transfer counts take the last fixture's value;
fm_ columns with "rating" in the name are averaged; everything else takes the first value.
n_fixtures says how many fixtures the gameweek had.

INSTALL: pip install pandas pyarrow

USAGE:
    python ingest.py
"""

import difflib
import glob
import os
import re
import unicodedata

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
FOTMOB_GLOB = os.path.join(BASE_DIR, "data", "raw", "fotmob", "*", "mw*", "player_stats.parquet")

PLAYERS_PATH = os.path.join(PROC_DIR, "fpl_players.parquet")
HISTORY_PATH = os.path.join(PROC_DIR, "fpl_player_history.parquet")
ID_MAP_PATH = os.path.join(PROC_DIR, "id_map.csv")

# FPL short club names -> the form FotMob uses. Anything else is matched automatically.
TEAM_ALIASES = {
    "man city": "manchester city",
    "man utd": "manchester united",
    "spurs": "tottenham hotspur",
    "nott m forest": "nottingham forest",
    "wolves": "wolverhampton wanderers",
}

STATE_COLUMNS = {"price", "selected", "transfers_balance", "transfers_in", "transfers_out"}
FM_IDENTITY = {
    "matchweek": "fm_matchweek", "player_name": "fm_player_name", "team": "fm_team",
    "opponent": "fm_opponent", "positionId": "fm_position_id", "match_id": "fm_match_id",
}
FUZZY_CUTOFF = 0.85

_SPECIAL = str.maketrans({"ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L", "ß": "ss", "æ": "ae", "Æ": "AE"})


# --------------------------------------------------------------------------- helpers

def norm(s):
    """Lowercase, strip accents and punctuation, so 'Ødegaard' and 'Odegaard' compare equal."""
    s = unicodedata.normalize("NFKD", str(s).translate(_SPECIAL)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s.replace("&", " and "))
    return re.sub(r"\s+", " ", s).strip()


def parquet_safe(df, text_cols=()):
    """Numeric where possible, text otherwise, so mixed columns never break to_parquet."""
    df = df.copy()
    for col in df.columns:
        if col in text_cols:
            df[col] = df[col].astype("string")
            continue
        if pd.api.types.is_bool_dtype(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].replace("", pd.NA)
        try:
            df[col] = pd.to_numeric(series)
        except (ValueError, TypeError):
            df[col] = series.astype("string")
    return df


def build_team_map(fpl_teams, fm_teams):
    """FPL club name -> FotMob club name. Teams that can't be matched are left out."""
    fm_norm = {norm(t): t for t in fm_teams}
    mapping = {}
    for team in fpl_teams:
        n = norm(team)
        n = norm(TEAM_ALIASES.get(n, n))
        if n in fm_norm:
            mapping[team] = fm_norm[n]
            continue
        contains = [orig for nn, orig in fm_norm.items() if n in nn or nn in n]
        if len(contains) == 1:
            mapping[team] = contains[0]
            continue
        close = difflib.get_close_matches(n, list(fm_norm), n=1, cutoff=0.6)
        if close:
            mapping[team] = fm_norm[close[0]]
    return mapping


# --------------------------------------------------------------------------- id mapping

def match_player(fm_name, fpl_team, index):
    """Returns (fpl_id, method, score) or (None, 'unmatched', 0). index = FPL player dicts."""
    n = norm(fm_name)
    tokens = set(n.split())
    same_team = [p for p in index if fpl_team and p["team"] == fpl_team]

    hits = [p for p in same_team if p["full"] == n or p["web"] == n]
    if len(hits) == 1:
        return hits[0]["id"], "exact_team", 1.0

    hits = [p for p in index if p["full"] == n]
    if len(hits) == 1:
        return hits[0]["id"], "exact_league", 1.0

    def web_in_name(web):  # "salah" in "mohamed salah"; "b fernandes" ~ "bruno fernandes"
        return bool(web) and all(
            any(t.startswith(w) for t in tokens) if len(w) == 1 else w in tokens for w in web.split()
        )

    hits = [p for p in same_team if web_in_name(p["web"])]
    if len(hits) == 1:
        return hits[0]["id"], "web_name_in_name", 1.0

    hits = [p for p in same_team if tokens <= set(p["full"].split())]  # "bruno fernandes" in the long full name
    if len(hits) == 1:
        return hits[0]["id"], "name_in_full_name", 1.0

    scored = sorted(
        ((max(difflib.SequenceMatcher(None, n, p["full"]).ratio(),
              difflib.SequenceMatcher(None, n, p["web"]).ratio()), p) for p in same_team),
        key=lambda x: -x[0],
    )
    if scored and scored[0][0] >= FUZZY_CUTOFF and (len(scored) == 1 or scored[1][0] < scored[0][0] - 0.05):
        return scored[0][1]["id"], "fuzzy_team", round(scored[0][0], 3)

    return None, "unmatched", 0.0


def build_id_map(fm, players, team_map):
    """Extends id_map.csv with any FotMob players not yet in it. Manual rows are kept."""
    index = [
        {"id": r.id, "full": norm(f"{r.first_name} {r.second_name}"), "web": norm(r.web_name), "team": r.team}
        for r in players.itertuples()
    ]
    fpl_info = players.set_index("id")[["web_name", "team"]]
    reverse_team = {v: k for k, v in team_map.items()}

    existing = pd.read_csv(ID_MAP_PATH) if os.path.exists(ID_MAP_PATH) else pd.DataFrame(
        columns=["fotmob_name", "fotmob_team", "fpl_id", "method", "score", "fpl_web_name", "fpl_team"]
    )
    kept = existing[existing["fpl_id"].notna()]
    done = set(zip(kept["fotmob_name"], kept["fotmob_team"]))

    new_rows = []
    for name, team in fm[["player_name", "team"]].drop_duplicates().itertuples(index=False):
        if (name, team) in done:
            continue
        pid, method, score = match_player(name, reverse_team.get(team), index)
        row = {"fotmob_name": name, "fotmob_team": team, "fpl_id": pid, "method": method, "score": score}
        if pid is not None:
            row["fpl_web_name"] = fpl_info.loc[pid, "web_name"]
            row["fpl_team"] = fpl_info.loc[pid, "team"]
        new_rows.append(row)

    id_map = pd.concat([kept, pd.DataFrame(new_rows)], ignore_index=True)
    id_map["fpl_id"] = pd.to_numeric(id_map["fpl_id"]).astype("Int64")
    id_map = id_map.sort_values(["fpl_id", "fotmob_name"], na_position="first").reset_index(drop=True)
    id_map.to_csv(ID_MAP_PATH, index=False, encoding="utf-8")
    return id_map


# --------------------------------------------------------------------------- joining

def load_fotmob():
    files = sorted(glob.glob(FOTMOB_GLOB))
    if not files:
        return None
    fm = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return parquet_safe(fm, text_cols=("player_name", "team", "opponent", "venue"))


def prepare_fotmob(fm, id_map):
    """Adds fpl_id, renames columns with an fm_ prefix, drops duplicate fixtures."""
    lookup = id_map.dropna(subset=["fpl_id"]).set_index(["fotmob_name", "fotmob_team"])["fpl_id"]
    keys = pd.MultiIndex.from_frame(fm[["player_name", "team"]])
    fm = fm.assign(fpl_id=lookup.reindex(keys).to_numpy())
    fm = fm.dropna(subset=["fpl_id"]).copy()
    fm["fpl_id"] = fm["fpl_id"].astype(int)

    renamed = {}
    for col in fm.columns:
        if col in ("fpl_id", "venue"):
            continue
        renamed[col] = FM_IDENTITY.get(col) or "fm_" + re.sub(r"\W+", "_", col.lower()).strip("_")
    fm = fm.rename(columns=renamed)
    fm = fm.loc[:, ~fm.columns.duplicated()]
    # same player + opponent + venue twice means a stale or duplicated scrape: keep the latest
    return fm.sort_values("fm_matchweek").drop_duplicates(["fpl_id", "fm_opponent", "venue"], keep="last")


def to_gameweek(df):
    key = ["player_id", "gameweek"]
    groups = df.groupby(key, sort=False)
    sum_cols, last_cols, mean_cols, first_cols, any_cols = [], [], [], [], []
    for c in df.columns:
        if c in key:
            continue
        s = df[c]
        if c == "has_fotmob":
            any_cols.append(c)
        elif c in STATE_COLUMNS:
            last_cols.append(c)
        elif c.startswith("fm_") and "rating" in c and pd.api.types.is_numeric_dtype(s):
            mean_cols.append(c)
        elif pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s) or c in ("fixture_id", "fm_match_id"):
            first_cols.append(c)
        else:
            sum_cols.append(c)

    parts = [
        groups[sum_cols].sum(min_count=1),
        groups[last_cols].last(),
        groups[mean_cols].mean(),
        groups[first_cols].first(),
        groups[any_cols].max(),
    ]
    out = pd.concat([p for p in parts if p.shape[1]], axis=1)
    out["n_fixtures"] = groups.size()
    out = out.reset_index()
    return out[[c for c in df.columns if c in out.columns] + ["n_fixtures"]]


def main():
    for path, hint in ((PLAYERS_PATH, "fpl.py"), (HISTORY_PATH, "fpl_history.py")):
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path} - run {hint} first.")

    players = pd.read_parquet(PLAYERS_PATH)
    base = pd.read_parquet(HISTORY_PATH)
    base["venue"] = base["was_home"].map({True: "home", False: "away"}).astype("string")
    print(f"FPL history: {len(base)} player-fixture rows, gameweeks {int(base['gameweek'].min())}-{int(base['gameweek'].max())}.")

    fm = load_fotmob()
    if fm is None:
        print("No FotMob files found - writing FPL-only tables.")
        merged = base
    else:
        fpl_teams = sorted(set(players["team"].dropna()) | set(base["opponent"].dropna()))
        fm_teams = sorted(set(fm["team"].dropna()) | set(fm["opponent"].dropna()))
        team_map = build_team_map(fpl_teams, fm_teams)
        missing = [t for t in fpl_teams if t not in team_map]
        if missing:
            print(f"[warn] no FotMob club found for FPL clubs: {missing}. Add them to TEAM_ALIASES.")

        os.makedirs(PROC_DIR, exist_ok=True)
        id_map = build_id_map(fm, players, team_map)
        unmatched = id_map[id_map["fpl_id"].isna()]
        print(f"ID map: {len(id_map) - len(unmatched)} FotMob players matched, {len(unmatched)} unmatched -> {ID_MAP_PATH}")
        if len(unmatched):
            print("  Fill in fpl_id for those rows (method stays 'unmatched', that's fine) and re-run.")

        fm_ready = prepare_fotmob(fm, id_map)
        base["fm_opponent"] = base["opponent"].map(team_map)
        merged = base.merge(
            fm_ready.drop(columns=["fm_matchweek", "fm_team"], errors="ignore"),
            left_on=["player_id", "fm_opponent", "venue"],
            right_on=["fpl_id", "fm_opponent", "venue"],
            how="left",
        ).drop(columns=["fpl_id"])
        merged["has_fotmob"] = merged["fm_match_id"].notna()

        scraped = set(fm["matchweek"].dropna().astype(int))
        played = merged[(merged["minutes"] > 0) & merged["gameweek"].isin(scraped)]
        if len(played):
            print(f"FotMob coverage on played rows in scraped gameweeks {sorted(scraped)}: "
                  f"{played['has_fotmob'].mean():.0%}")

    merged = merged.drop(columns=["venue"]).copy()
    merged = parquet_safe(merged, text_cols=("web_name", "team", "position", "opponent", "kickoff_time"))
    gw = to_gameweek(merged)

    os.makedirs(PROC_DIR, exist_ok=True)
    merged.to_parquet(os.path.join(PROC_DIR, "player_fixture.parquet"), index=False)
    gw.to_parquet(os.path.join(PROC_DIR, "player_gw.parquet"), index=False)
    print(f"Wrote player_fixture.parquet ({len(merged)} rows) and player_gw.parquet ({len(gw)} rows) -> {PROC_DIR}")


if __name__ == "__main__":
    main()