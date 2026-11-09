"""
fotmob_scraper.py

Scrapes team-level and player-level match stats from FotMob's internal JSON API for
a given Premier League matchweek (2026-27 season), writing them into
raw/fotmob/2026-27/mwXX/ as CSVs, matching the format used by sofascore_scraper.py.

WHY THIS ONE LOOKS DIFFERENT FROM THE SOFASCORE SCRAPER:
FotMob now sits behind Cloudflare Turnstile (a CAPTCHA-style bot challenge). Neither
plain `requests` nor `cloudscraper` (which worked fine for Sofascore) can solve this -
it genuinely requires a real, automated browser to load the page and let the challenge
resolve before the JSON becomes available. This script uses `nodriver`, an
undetected-Chrome automation library, for that reason. This is heavier than the
Sofascore approach: it launches an actual Chrome browser process, needs a ~5 second
warm-up on FotMob's homepage before making any API calls, and each request involves
loading a real page rather than a lightweight HTTP call. Expect a full matchweek scrape
to take a few minutes, not seconds.

INSTALL: pip install nodriver   (needs real Chrome/Chromium installed on your machine -
you already have this if you use Chrome as a browser day-to-day)

VERIFICATION STATUS (as of build time):
- League id 47 = Premier League: CONFIRMED via multiple independent sources.
- /leagues?id=47&season=... -> {"matches": {"allMatches": [...]}}: CONFIRMED via a
  working scraper's source code (exact param names and response shape).
- /matchDetails?matchId=... -> content.stats.Periods.All.stats -> team-level stats:
  CONFIRMED via a working scraper's source code that parses this exact path.
- Player-level stats (content.lineup / content.playerStats): NOT independently
  confirmed at the field level. I could only confirm content.playerStats is a dict
  keyed by player id containing at least a 'name' field - the rest of the per-player
  stat fields are my best-effort inference, same situation the Sofascore lineups
  endpoint was in before we test-ran it (which turned out correct, but this one
  genuinely could differ). Always check raw_json/*_matchdetails.json against what
  player_stats.csv produces before trusting a full matchweek unattended.

RATE LIMITING: each match needs 1 page load for details. Delay between matches is
larger than the Sofascore scraper's, since we're driving a real browser, not just
sending HTTP requests - don't shrink this without reason.
"""

import argparse
import csv
import json
import os
import re
import asyncio
import nodriver as uc

LEAGUE_ID = 47  # Premier League - confirmed via fotmob.com/leagues/47/overview/premier-league
REQUEST_DELAY_SECONDS = 3
WARMUP_WAIT_SECONDS = 5
MAX_RETRIES = 3

TEAM_COLUMN_PRIORITY = ["team", "opponent", "venue", "match_id"]
PLAYER_COLUMN_PRIORITY = [
    "player_name", "team", "opponent", "venue", "positionId", "match_id",
]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "raw", "fotmob", "2026-27")


def ordered_fieldnames(rows, priority_columns):
    all_keys = {k for row in rows for k in row.keys()}
    ordered = [k for k in priority_columns if k in all_keys]
    remaining = sorted(all_keys - set(ordered))
    return ordered + remaining


async def fetch_json(browser, path, max_retries=MAX_RETRIES):
    """
    Loads a FotMob API path through the real browser and extracts the JSON body.
    FotMob's API returns raw JSON with no HTML wrapper, but the browser renders it
    inside a <pre> tag - same trick every current FotMob scraper relies on to get past
    Cloudflare's Turnstile challenge (a plain HTTP client never gets the challenge
    solved for it).
    """
    url = f"https://www.fotmob.com/api/data/{path}"
    for attempt in range(1, max_retries + 1):
        try:
            page = await browser.get(url)
            await asyncio.sleep(2)  # let the page fully render
            content = await page.get_content()
            match = re.search(r"<pre[^>]*>(.*?)</pre>", content, re.DOTALL)
            if not match:
                wait_s = WARMUP_WAIT_SECONDS * attempt
                print(f"  [warn] no JSON found for {path} (challenge screen?), waiting {wait_s}s (attempt {attempt}/{max_retries})")
                await asyncio.sleep(wait_s)
                continue
            return json.loads(match.group(1).strip())
        except Exception as e:
            print(f"  [warn] {path} -> {e} (attempt {attempt}/{max_retries})")
            await asyncio.sleep(REQUEST_DELAY_SECONDS * attempt)
    print(f"  [error] giving up on {path}")
    return None


async def get_season_matches(browser, season="2026/2027", debug_dump_path=None):
    season_encoded = season.replace("/", "%2F")
    path = f"leagues?id={LEAGUE_ID}&ccode3=ENG&season={season_encoded}"
    data = await fetch_json(browser, path)

    if debug_dump_path:
        with open(debug_dump_path, "w") as f:
            json.dump(data, f, indent=2)

    if not data:
        print("  [debug] fetch_json returned nothing at all for the leagues endpoint.")
        return []

    matches_block = data.get("matches")
    if matches_block is not None:
        all_matches = matches_block.get("allMatches")
        if all_matches is not None:
            return all_matches

    # "matches.allMatches" wasn't there - try "fixtures" instead (confirmed as the
    # real top-level key for this endpoint from a live test run).
    fixtures_block = data.get("fixtures")
    if isinstance(fixtures_block, list):
        return fixtures_block
    if isinstance(fixtures_block, dict):
        for key in ("allMatches", "matches", "data"):
            if isinstance(fixtures_block.get(key), list):
                return fixtures_block[key]
        print(f"  [debug] 'fixtures' is a dict but none of the expected keys were lists. Keys were: {list(fixtures_block.keys())}")
        return []

    print(f"  [debug] no usable 'matches' or 'fixtures' key found. Top-level keys were: {list(data.keys())}")
    return []


def extract_team_stats(match_json, home_name, away_name, match_id):
    """
    Flattens content.stats.Periods.All.stats into one row for home, one for away.
    Confirmed structure (from working scraper source): a list of stat groups, each
    with a 'stats' list of items; each item has a label (title/key) and a 'stats'
    field holding [home_value, away_value].
    """
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


def extract_player_stats(match_json, home_name, away_name, match_id):
    """
    UNVERIFIED at the field level - see module docstring. Pulls from
    content.lineup (home/away player lists with side + position) cross-referenced
    with content.playerStats (dict of playerId -> stat dict) where available.
    Defensive .get() usage throughout so a shape mismatch produces missing fields
    rather than a crash - and the raw JSON is always saved alongside this, so nothing
    is lost even if this flattening logic needs fixing once you see real output.
    """
    rows = []
    content = match_json.get("content", {}) or {}
    player_stats_lookup = content.get("playerStats") or {}
    lineup = content.get("lineup") or {}

    for side_key, team_name, opp_name in (
        ("homeTeam", home_name, away_name),
        ("awayTeam", away_name, home_name),
    ):
        side_data = lineup.get(side_key, {}) or {}

        # Confirmed from real output: "starters"/"players" alone only returns the
        # starting XI - substitutes live under a separate key. Gather from every
        # plausible key and dedupe by player id, rather than picking just one.
        raw_lists = []
        for key in ("starters", "bench", "subs", "substitutes", "players"):
            val = side_data.get(key)
            if isinstance(val, list):
                raw_lists.append(val)

        seen_ids = set()
        flat_players = []
        for players_list in raw_lists:
            for p in players_list:
                # some of these come as rows-of-formation-lines (nested lists) rather
                # than a flat list of player dicts - unwrap either shape
                candidates = p if isinstance(p, list) else [p]
                for cand in candidates:
                    if not isinstance(cand, dict):
                        continue
                    pid = cand.get("id", cand.get("playerId"))
                    if pid is not None and pid in seen_ids:
                        continue
                    if pid is not None:
                        seen_ids.add(pid)
                    flat_players.append(cand)

        for p in flat_players:
            player_id = str(p.get("id", p.get("playerId", "")))
            stat_detail = player_stats_lookup.get(player_id, {}) if player_id else {}
            row = {
                "player_name": p.get("name", stat_detail.get("name", "")),
                "team": team_name,
                "opponent": opp_name,
                "venue": "home" if side_key == "homeTeam" else "away",
                "positionId": p.get("positionId", ""),
                "match_id": match_id,
            }
            stats = stat_detail.get("stats", {}) if isinstance(stat_detail, dict) else {}
            if isinstance(stats, list):
                stat_dict = flatten_player_stat_groups(stats)
            elif isinstance(stats, dict):
                stat_dict = stats
            else:
                stat_dict = {}
            for k, v in stat_dict.items():
                row[k] = v
            rows.append(row)

    return rows


def flatten_player_stat_groups(stats_list):
    """
    FotMob's real per-player stats shape (confirmed from a live test run): a list of
    category groups (Attack, Defense, Duels, Top stats), each with its OWN nested dict
    of individual stats, e.g.:
        {"title": "Attack", "stats": {"Touches": {"stat": {"value": 28}}, ...}}
    This drills into that second layer to produce clean, individual numeric columns
    (e.g. "Touches": 28) instead of the raw nested JSON. For fraction-style stats that
    have both a value and a total (e.g. "Accurate passes": 33 of 38), the total gets
    its own "<stat name> attempted" column alongside it.
    """
    flat = {}
    for group in stats_list:
        if not isinstance(group, dict):
            continue
        inner_stats = group.get("stats")
        if not isinstance(inner_stats, dict):
            continue
        for stat_name, stat_info in inner_stats.items():
            if not isinstance(stat_info, dict):
                continue
            stat_obj = stat_info.get("stat", {}) or {}
            value = stat_obj.get("value")
            total = stat_obj.get("total")
            if value is not None:
                flat[stat_name] = value
            if total is not None:
                flat[f"{stat_name} attempted"] = total
    return flat


async def scrape_matchweek(matchweek, season="2026/2027"):
    mw_str = f"mw{matchweek:02d}"
    out_dir = os.path.join(RAW_DIR, mw_str)
    raw_json_dir = os.path.join(out_dir, "raw_json")
    os.makedirs(raw_json_dir, exist_ok=True)

    print("Starting browser and warming up on fotmob.com (needed for Cloudflare)...")
    browser = await uc.start()
    team_rows = []
    player_rows = []

    try:
        await browser.get("https://www.fotmob.com")
        await asyncio.sleep(WARMUP_WAIT_SECONDS)
        print("Browser ready. Fetching season fixture list...")

        all_matches = await get_season_matches(
            browser, season, debug_dump_path=os.path.join(raw_json_dir, "leagues_response_debug.json")
        )

        # Diagnostic dump - always save this so we can see the real shape if filtering fails
        with open(os.path.join(raw_json_dir, "season_matches_debug.json"), "w") as f:
            json.dump(all_matches, f, indent=2)

        print(f"Fetched {len(all_matches)} total matches for the season.")
        if all_matches:
            sample_keys = list(all_matches[0].keys())
            print(f"Keys on a sample match: {sample_keys}")
            round_values = sorted({m.get("round") for m in all_matches[:20]}, key=lambda x: (x is None, x))
            print(f"Sample of 'round' values seen: {round_values}")

        mw_matches = [m for m in all_matches if str(m.get("round")) == str(matchweek)]

        if not mw_matches:
            print(f"No matches found for matchweek {matchweek}. Has FotMob published this round's fixtures?")
            return

        for m in mw_matches:
            home_name = m.get("home", {}).get("name", "UNKNOWN_HOME")
            away_name = m.get("away", {}).get("name", "UNKNOWN_AWAY")
            match_id = m.get("id")
            finished = m.get("status", {}).get("finished")

            if not finished:
                print(f"  Skipping {home_name} vs {away_name} (not finished yet)")
                continue

            print(f"  Scraping {home_name} vs {away_name} (match {match_id})...")
            details = await fetch_json(browser, f"matchDetails?matchId={match_id}")
            await asyncio.sleep(REQUEST_DELAY_SECONDS)

            if not details:
                print(f"  [warn] no match details returned for {match_id}, skipping")
                continue

            with open(os.path.join(raw_json_dir, f"{match_id}_matchdetails.json"), "w") as f:
                json.dump(details, f, indent=2)

            home_row, away_row = extract_team_stats(details, home_name, away_name, match_id)
            team_rows.extend([home_row, away_row])

            player_rows.extend(extract_player_stats(details, home_name, away_name, match_id))
    finally:
        # ALWAYS close the browser, even if something above crashes or raises. Leaving
        # this out is what caused the "Session with given id not found" error on the
        # next run - a crashed run was leaving an orphaned Chrome process behind.
        browser.stop()
        print("Browser closed.")

    if team_rows:
        team_fields = ordered_fieldnames(team_rows, TEAM_COLUMN_PRIORITY)
        with open(os.path.join(out_dir, "team_stats.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=team_fields)
            writer.writeheader()
            writer.writerows(team_rows)
        print(f"Wrote {len(team_rows)} team rows -> {out_dir}/team_stats.csv")

    if player_rows:
        player_fields = ordered_fieldnames(player_rows, PLAYER_COLUMN_PRIORITY)
        with open(os.path.join(out_dir, "player_stats.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=player_fields)
            writer.writeheader()
            writer.writerows(player_rows)
        print(f"Wrote {len(player_rows)} player rows -> {out_dir}/player_stats.csv")
    else:
        print("No player rows written - check raw_json/*_matchdetails.json to see the real lineup/playerStats shape.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape a PL matchweek from FotMob.")
    parser.add_argument("--matchweek", type=int, required=True, help="Matchweek number (1-38)")
    parser.add_argument("--season", default="2026/2027", help="Season string, e.g. '2026/2027'")
    args = parser.parse_args()
    uc.loop().run_until_complete(scrape_matchweek(args.matchweek, args.season))
