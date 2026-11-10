"""
fotmob_scraper.py (v2)

Scrapes team-level and player-level match stats for a Premier League matchweek from
FotMob's internal JSON API. Writes to data/raw/fotmob/<season>/mwXX/:

    raw_json/<match_id>_matchdetails.json   raw API response, one per match (the cache)
    matches.json                            the finished matches for that week
    team_stats.parquet                      one row per team per match
    player_stats.parquet                    one row per player per match

HOW IT WORKS (two separate steps):
  1. SCRAPE: open ONE real Chrome window (FotMob sits behind Cloudflare Turnstile, so
     plain HTTP clients are blocked), warm up on fotmob.com, then call the API with
     fetch() from inside that page. Each response is saved as raw JSON. Matches whose
     raw JSON is already on disk are skipped.
  2. PARSE: flatten the saved raw JSON into the two tables. This needs no browser and no
     network, so if the parsing breaks you fix it and re-run with --parse-only.

If fetch() from inside the page is ever blocked, the client falls back to loading the
API URL as a page and reading the JSON out of the <pre> tag (the old method).

VERIFICATION STATUS: the fetch()-in-page approach is the same one the Sofascore scraper
uses, but it is UNTESTED against FotMob. Player-level field names are still
best-effort, so compare player_stats.parquet against one raw JSON file before trusting it.

INSTALL: pip install nodriver pandas pyarrow   (also needs Chrome installed)

USAGE:
    python fotmob_scraper.py                       # auto-select the current matchweek
    python fotmob_scraper.py --matchweek 10
    python fotmob_scraper.py --matchweek 10 --refresh     # re-download cached matches
    python fotmob_scraper.py --parse-only                 # re-parse saved JSON, no browser
"""

import argparse
import asyncio
import html
import json
import os
import re

import nodriver as uc
import pandas as pd

LEAGUE_ID = 47  # Premier League
API = "https://www.fotmob.com/api/data"
REQUEST_DELAY_SECONDS = 1.5
WARMUP_WAIT_SECONDS = 5
MAX_RETRIES = 3

TEAM_COLUMN_PRIORITY = ["matchweek", "team", "opponent", "venue", "match_id"]
PLAYER_COLUMN_PRIORITY = [
    "matchweek", "player_name", "team", "opponent", "venue", "positionId", "match_id",
]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def season_dir(season):
    """'2026/2027' -> <repo>/data/raw/fotmob/2026-27"""
    start, end = season.split("/")
    return os.path.join(BASE_DIR, "data", "raw", "fotmob", f"{start}-{end[-2:]}")


# --------------------------------------------------------------------------- scraping

class FotmobClient:
    """One browser, one warmed-up page, many API calls."""

    def __init__(self):
        self.browser = None
        self.page = None

    async def start(self):
        print("Starting browser and warming up on fotmob.com (needed for Cloudflare)...")
        self.browser = await uc.start()
        self.page = await self.browser.get("https://www.fotmob.com")
        await asyncio.sleep(WARMUP_WAIT_SECONDS)
        print("Browser ready.")

    def stop(self):
        # Always called from a finally block so a crash never leaves Chrome running.
        if self.browser is not None:
            try:
                self.browser.stop()
            except Exception:
                pass
            self.browser = None
            print("Browser closed.")

    async def _fetch_in_page(self, url):
        js = (
            f'fetch("{url}", {{credentials: "include"}})'
            ".then(async r => JSON.stringify({status: r.status, body: await r.text()}))"
            ".catch(e => JSON.stringify({status: 0, body: String(e)}))"
        )
        res = await self.page.evaluate(js, await_promise=True)
        return json.loads(res)

    async def _fetch_by_navigation(self, url):
        self.page = await self.browser.get(url)
        await asyncio.sleep(2)
        content = await self.page.get_content()
        match = re.search(r"<pre[^>]*>(.*?)</pre>", content, re.DOTALL)
        if not match:
            return None
        return json.loads(html.unescape(match.group(1)).strip())

    async def get_json(self, path):
        url = f"{API}/{path}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = await self._fetch_in_page(url)
                if result.get("status") == 200:
                    return json.loads(result["body"])
                print(f"  [warn] {path} -> HTTP {result.get('status')} (attempt {attempt}/{MAX_RETRIES})")
                data = await self._fetch_by_navigation(url)  # fallback: old method
                if data is not None:
                    return data
            except Exception as e:
                print(f"  [warn] {path} -> {e} (attempt {attempt}/{MAX_RETRIES})")
            await asyncio.sleep(WARMUP_WAIT_SECONDS * attempt)
        print(f"  [error] giving up on {path}")
        return None


async def get_season_matches(client, season):
    season_encoded = season.replace("/", "%2F")
    data = await client.get_json(f"leagues?id={LEAGUE_ID}&ccode3=ENG&season={season_encoded}")
    if not data:
        return []

    all_matches = (data.get("matches") or {}).get("allMatches")
    if isinstance(all_matches, list):
        return all_matches

    fixtures = data.get("fixtures")
    if isinstance(fixtures, list):
        return fixtures
    if isinstance(fixtures, dict):
        for key in ("allMatches", "matches", "data"):
            if isinstance(fixtures.get(key), list):
                return fixtures[key]

    print(f"  [debug] no usable fixture list. Top-level keys were: {list(data.keys())}")
    return []


def detect_current_matchweek(all_matches):
    """Highest round with at least one finished match. None if nothing has finished."""
    rounds = []
    for m in all_matches:
        if (m.get("status") or {}).get("finished"):
            try:
                rounds.append(int(m.get("round")))
            except (TypeError, ValueError):
                continue
    return max(rounds) if rounds else None


async def scrape(matchweek, season, refresh):
    """Downloads raw JSON for the matchweek. Returns the matchweek number used."""
    season_path = season_dir(season)
    os.makedirs(season_path, exist_ok=True)

    client = FotmobClient()
    await client.start()
    try:
        print("Fetching season fixture list...")
        all_matches = await get_season_matches(client, season)
        with open(os.path.join(season_path, "season_matches.json"), "w", encoding="utf-8") as f:
            json.dump(all_matches, f, indent=2)
        print(f"Fetched {len(all_matches)} total matches for the season.")

        if matchweek is None:
            matchweek = detect_current_matchweek(all_matches)
            if matchweek is None:
                print("No finished matches yet this season - nothing to scrape.")
                return None
            print(f"Auto-selected matchweek {matchweek} (latest round with a finished match).")

        mw_dir = os.path.join(season_path, f"mw{matchweek:02d}")
        raw_dir = os.path.join(mw_dir, "raw_json")
        os.makedirs(raw_dir, exist_ok=True)

        week = [m for m in all_matches if str(m.get("round")) == str(matchweek)]
        if not week:
            print(f"No matches found for matchweek {matchweek}.")
            return None

        finished = []
        for m in week:
            home = (m.get("home") or {}).get("name", "UNKNOWN_HOME")
            away = (m.get("away") or {}).get("name", "UNKNOWN_AWAY")
            if not (m.get("status") or {}).get("finished"):
                print(f"  Skipping {home} vs {away} (not finished yet)")
                continue
            finished.append({"match_id": m.get("id"), "home": html.unescape(home), "away": html.unescape(away)})

        with open(os.path.join(mw_dir, "matches.json"), "w", encoding="utf-8") as f:
            json.dump(finished, f, indent=2)

        for m in finished:
            raw_path = os.path.join(raw_dir, f"{m['match_id']}_matchdetails.json")
            label = f"{m['home']} vs {m['away']} (match {m['match_id']})"
            if os.path.exists(raw_path) and not refresh:
                print(f"  Cached: {label}")
                continue
            print(f"  Scraping {label}...")
            details = await client.get_json(f"matchDetails?matchId={m['match_id']}")
            if not details:
                print("  [warn] no match details returned, skipping")
                continue
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(details, f, indent=2)
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
    finally:
        client.stop()

    return matchweek


# --------------------------------------------------------------------------- parsing

def extract_team_stats(match_json, home_name, away_name, match_id):
    """content.stats.Periods.All.stats -> one row for home, one for away."""
    home_row = {"team": home_name, "opponent": away_name, "venue": "home", "match_id": match_id}
    away_row = {"team": away_name, "opponent": home_name, "venue": "away", "match_id": match_id}

    try:
        groups = match_json["content"]["stats"]["Periods"]["All"]["stats"]
    except (KeyError, TypeError):
        return home_row, away_row

    for group in groups:
        for item in group.get("stats", []):
            label = item.get("title") or item.get("key") or "unknown_stat"
            values = item.get("stats")
            if isinstance(values, list) and len(values) == 2:
                home_row[label] = values[0]
                away_row[label] = values[1]
    return home_row, away_row


def flatten_player_stat_groups(stats_list):
    """
    FotMob's per-player stats are a list of category groups (Attack, Defense, Duels...),
    each with its own dict of stats, e.g. {"Touches": {"stat": {"value": 28}}}. This turns
    that into flat columns ("Touches": 28). Fraction-style stats also get a
    "<name> attempted" column for the total.
    """
    flat = {}
    for group in stats_list:
        if not isinstance(group, dict):
            continue
        inner = group.get("stats")
        if not isinstance(inner, dict):
            continue
        for stat_name, stat_info in inner.items():
            if not isinstance(stat_info, dict):
                continue
            stat_obj = stat_info.get("stat", {}) or {}
            if stat_obj.get("value") is not None:
                flat[stat_name] = stat_obj["value"]
            if stat_obj.get("total") is not None:
                flat[f"{stat_name} attempted"] = stat_obj["total"]
    return flat


def extract_player_stats(match_json, home_name, away_name, match_id):
    """
    UNVERIFIED at the field level. Reads content.lineup (who played, position) and joins
    content.playerStats (stats by player id). Defensive .get() throughout, so a shape
    mismatch gives missing columns rather than a crash.
    """
    rows = []
    content = match_json.get("content", {}) or {}
    stats_lookup = content.get("playerStats") or {}
    lineup = content.get("lineup") or {}

    for side_key, team, opp in (("homeTeam", home_name, away_name), ("awayTeam", away_name, home_name)):
        side = lineup.get(side_key, {}) or {}

        # Starters and subs live under different keys; gather all and dedupe by player id.
        seen, players = set(), []
        for key in ("starters", "bench", "subs", "substitutes", "players"):
            val = side.get(key)
            if not isinstance(val, list):
                continue
            for p in val:
                for cand in (p if isinstance(p, list) else [p]):
                    if not isinstance(cand, dict):
                        continue
                    pid = cand.get("id", cand.get("playerId"))
                    if pid is not None:
                        if pid in seen:
                            continue
                        seen.add(pid)
                    players.append(cand)

        for p in players:
            pid = str(p.get("id", p.get("playerId", "")))
            detail = stats_lookup.get(pid, {}) if pid else {}
            row = {
                "player_name": p.get("name", detail.get("name", "") if isinstance(detail, dict) else ""),
                "team": team,
                "opponent": opp,
                "venue": "home" if side_key == "homeTeam" else "away",
                "positionId": p.get("positionId", ""),
                "match_id": match_id,
            }
            stats = detail.get("stats", {}) if isinstance(detail, dict) else {}
            if isinstance(stats, list):
                row.update(flatten_player_stat_groups(stats))
            elif isinstance(stats, dict):
                row.update(stats)
            rows.append(row)
    return rows


TEXT_COLUMNS = ("team", "opponent", "venue", "player_name")


def to_table(rows, priority_columns):
    """List of dicts -> DataFrame with identity columns first and parquet-safe dtypes."""
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col in TEXT_COLUMNS:
            df[col] = df[col].astype("string")
            continue
        series = df[col].replace("", pd.NA)  # blanks become missing, not text
        try:
            df[col] = pd.to_numeric(series)
        except (ValueError, TypeError):
            df[col] = series.astype("string")  # genuinely mixed -> text, so parquet won't fail
    ordered = [c for c in priority_columns if c in df.columns]
    rest = sorted(set(df.columns) - set(ordered))
    return df[ordered + rest]


def parse_matchweek(matchweek, season):
    """Builds the two tables from saved raw JSON. No browser, no network."""
    mw_dir = os.path.join(season_dir(season), f"mw{matchweek:02d}")
    matches_path = os.path.join(mw_dir, "matches.json")
    if not os.path.exists(matches_path):
        print(f"No matches.json in {mw_dir} - run a scrape for this matchweek first.")
        return

    with open(matches_path, encoding="utf-8") as f:
        matches = json.load(f)

    team_rows, player_rows = [], []
    for m in matches:
        raw_path = os.path.join(mw_dir, "raw_json", f"{m['match_id']}_matchdetails.json")
        if not os.path.exists(raw_path):
            print(f"  [warn] no raw JSON for match {m['match_id']}, skipping")
            continue
        with open(raw_path, encoding="utf-8") as f:
            details = json.load(f)

        home_row, away_row = extract_team_stats(details, m["home"], m["away"], m["match_id"])
        team_rows += [home_row, away_row]
        player_rows += extract_player_stats(details, m["home"], m["away"], m["match_id"])

    for rows in (team_rows, player_rows):
        for r in rows:
            r["matchweek"] = matchweek

    if team_rows:
        path = os.path.join(mw_dir, "team_stats.parquet")
        to_table(team_rows, TEAM_COLUMN_PRIORITY).to_parquet(path, index=False)
        print(f"Wrote {len(team_rows)} team rows -> {path}")
    if player_rows:
        path = os.path.join(mw_dir, "player_stats.parquet")
        to_table(player_rows, PLAYER_COLUMN_PRIORITY).to_parquet(path, index=False)
        print(f"Wrote {len(player_rows)} player rows -> {path}")
    else:
        print("No player rows written - compare a raw_json file against extract_player_stats().")


def latest_local_matchweek(season):
    """Highest mwXX folder already on disk (used by --parse-only with no --matchweek)."""
    path = season_dir(season)
    if not os.path.isdir(path):
        return None
    weeks = [int(d[2:]) for d in os.listdir(path) if re.fullmatch(r"mw\d{2}", d)]
    return max(weeks) if weeks else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape a PL matchweek from FotMob.")
    parser.add_argument("--matchweek", type=int, default=None,
                        help="Matchweek number (1-38). Omit to auto-select the current matchweek.")
    parser.add_argument("--season", default="2026/2027", help="Season string, e.g. '2026/2027'")
    parser.add_argument("--refresh", action="store_true", help="Re-download matches already cached on disk")
    parser.add_argument("--parse-only", action="store_true", help="Skip scraping; rebuild tables from saved JSON")
    args = parser.parse_args()

    if args.parse_only:
        mw = args.matchweek or latest_local_matchweek(args.season)
    else:
        mw = uc.loop().run_until_complete(scrape(args.matchweek, args.season, args.refresh))

    if mw is not None:
        parse_matchweek(mw, args.season)
    else:
        print("Nothing to parse.")