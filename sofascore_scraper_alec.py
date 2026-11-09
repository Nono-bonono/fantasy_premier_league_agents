"""
sofascore_scraper.py

Scrapes team-level and player-level match stats from Sofascore's public JSON API
for a given Premier League matchweek (2026-27 season) and writes them into
raw/sofascore/2026-27/mwXX/ as CSVs, matching the fuzzy_match_players.py input format
(columns: team, player_name, ...stats).

VERIFICATION STATUS (as of build time):
- /event/{id}/statistics -> CONFIRMED working, live-tested against a real Sofascore
  response. Structure: {"statistics": [{"period": "ALL", "groups": [{"groupName":...,
  "statisticsItems": [{"name":..., "key":..., "home":..., "away":..., "homeValue":...,
  "awayValue":...}]}]}, {"period": "1ST", ...}, ...]}. We only use the "ALL" period.
- /event/{id}/lineups -> NOT live-tested (Sofascore isn't reachable from this sandbox's
  network, and the 2026-27 season has no finished matches yet to test against anyway).
  The parsing below is based on strong secondary confirmation from several independent
  open-source Sofascore scrapers, but you should run scrape_matchweek() against one
  real finished match early and manually check raw/.../lineups_raw/*.json against what
  player_stats.csv produces, BEFORE trusting this for a full matchweek unattended.

RATE LIMITING: this hits 3 endpoints per match (events, statistics, lineups) x 10
matches = ~30 requests per matchweek. REQUEST_DELAY_SECONDS below spaces these out -
don't set it to 0, Sofascore will start blocking your IP if you hammer it.

USAGE:
    python sofascore_scraper.py --matchweek 1
This assumes the folder layout we set up: raw/sofascore/2026-27/mwXX/ relative to
wherever pl_data_26_27/ lives. Adjust BASE_DIR below if you run it from elsewhere.
"""

import argparse
import csv
import json
import os
import time
import atexit
import nodriver as uc

BASE_URL = "https://www.sofascore.com/api/v1"  # same-origin host; api.sofascore.com is blocked (403)
TOURNAMENT_ID = 17  # Premier League, confirmed via search (sofascore.com/football/tournament/england/premier-league/17)
REQUEST_DELAY_SECONDS = 1.5

HEADERS = {
    # Sofascore's API is undocumented/unofficial and blocks plain `requests` calls at
    # the anti-bot layer even with correct headers (it inspects the TLS connection
    # itself, not just headers). We use cloudscraper below instead of raw requests to
    # get past this - if THAT stops working too, the next step up is real browser
    # automation (Playwright/Selenium), which several other Sofascore scrapers have had
    # to fall back to as Sofascore's protection has gotten stricter.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# --- Browser session --------------------------------------------------------------
# Sofascore now blocks every plain HTTP client (requests, cloudscraper, curl_cffi) with a
# 403/JS challenge. A real browser session passes, so we start ONE nodriver Chrome,
# load a Sofascore page once, and make every API call from inside that page via fetch().
_LOOP = None
_BROWSER = None
_PAGE = None
LANDING_URL = "https://www.sofascore.com/football/tournament/england/premier-league/17"


def _loop():
    global _LOOP
    if _LOOP is None:
        _LOOP = uc.loop()
    return _LOOP


async def _start_browser():
    global _BROWSER, _PAGE
    _BROWSER = await uc.start()
    _PAGE = await _BROWSER.get(LANDING_URL)
    await _PAGE.sleep(8)


async def _fetch(url):
    js = (
        f'fetch("{url}", {{credentials: "include"}})'
        '.then(async r => JSON.stringify({status: r.status, body: await r.text()}))'
        '.catch(e => JSON.stringify({status: 0, body: String(e)}))'
    )
    res = await _PAGE.evaluate(js, await_promise=True)
    return json.loads(res)


async def _refresh_page():
    global _PAGE
    _PAGE = await _BROWSER.get(LANDING_URL)
    await _PAGE.sleep(6)


def close_browser():
    """Shut the shared browser down. Registered with atexit, safe to call repeatedly."""
    global _BROWSER, _PAGE
    if _BROWSER is not None:
        try:
            _BROWSER.stop()
        except Exception:
            pass
    _BROWSER = None
    _PAGE = None


atexit.register(close_browser)


def _ensure_browser():
    if _BROWSER is None or _PAGE is None:
        _loop().run_until_complete(_start_browser())

# BASE_DIR should point at your pl_data_26_27 folder. Adjust if this script lives
# somewhere other than pl_data_26_27/scripts/scrapers/.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "raw", "sofascore", "2026-27")


TEAM_COLUMN_PRIORITY = ["team", "opponent", "venue", "event_id"]
PLAYER_COLUMN_PRIORITY = [
    "player_name", "team", "opponent", "venue", "position",
    "shirt_number", "substitute", "event_id",
]


def ordered_fieldnames(rows, priority_columns):
    """
    Returns column order for a CSV: the priority (identifying/qualitative) columns
    first in the given order, then every remaining column alphabetically.
    """
    all_keys = {k for row in rows for k in row.keys()}
    ordered = [k for k in priority_columns if k in all_keys]
    remaining = sorted(all_keys - set(ordered))
    return ordered + remaining


def request_json(url, max_retries=3):
    """GET a URL via the in-browser fetch and parse JSON, with retry/backoff. None on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            _ensure_browser()
            result = _loop().run_until_complete(_fetch(url))
            status = result.get("status")
            if status == 200:
                return json.loads(result["body"])
            elif status == 404:
                return None  # e.g. lineups not yet posted for an unplayed match
            else:
                print(f"  [warn] {url} -> HTTP {status} (attempt {attempt}/{max_retries})")
                if status in (0, 403):
                    # session may have gone stale - reload the landing page for fresh cookies
                    try:
                        _loop().run_until_complete(_refresh_page())
                    except Exception:
                        close_browser()
        except Exception as e:
            print(f"  [warn] {url} -> {e} (attempt {attempt}/{max_retries})")
            close_browser()  # force a clean restart on next attempt
        time.sleep(REQUEST_DELAY_SECONDS * attempt)
    print(f"  [error] giving up on {url}")
    return None


def get_season_id(target_year_label="26/27"):
    """Looks up the Sofascore season id for the given PL season label (e.g. '26/27')."""
    data = request_json(f"{BASE_URL}/unique-tournament/{TOURNAMENT_ID}/seasons")
    if not data:
        raise RuntimeError("Could not fetch season list from Sofascore.")
    for season in data.get("seasons", []):
        if season.get("year") == target_year_label:
            return season["id"]
    raise RuntimeError(
        f"Could not find a season labeled '{target_year_label}'. "
        f"Available: {[s.get('year') for s in data.get('seasons', [])]}"
    )


def get_events_for_round(season_id, round_number):
    """Returns the list of raw event dicts for a given PL matchweek (round)."""
    url = f"{BASE_URL}/unique-tournament/{TOURNAMENT_ID}/season/{season_id}/events/round/{round_number}"
    data = request_json(url)
    if not data:
        return []
    return data.get("events", [])


def get_event_statistics(event_id):
    """Returns a flat dict of {stat_key: {'home': val, 'away': val}} for the ALL period."""
    data = request_json(f"{BASE_URL}/event/{event_id}/statistics")
    if not data:
        return {}
    flat = {}
    for period_block in data.get("statistics", []):
        if period_block.get("period") != "ALL":
            continue
        for group in period_block.get("groups", []):
            for item in group.get("statisticsItems", []):
                key = item.get("key") or item.get("name")
                flat[key] = {
                    "home": item.get("homeValue", item.get("home")),
                    "away": item.get("awayValue", item.get("away")),
                }
    return flat


def get_event_lineups(event_id):
    """
    Returns list of player-stat dicts: [{'side': 'home'/'away', 'player_name': ...,
    'position': ..., stat_key: value, ...}, ...]
    UNVERIFIED against a live response - see module docstring. Defensive .get() usage
    throughout so a shape mismatch produces missing fields/empty rows rather than a crash.
    """
    data = request_json(f"{BASE_URL}/event/{event_id}/lineups")
    if not data:
        return []

    rows = []
    for side in ("home", "away"):
        side_data = data.get(side, {})
        for p in side_data.get("players", []):
            player_info = p.get("player", {})
            row = {
                "side": side,
                "player_name": player_info.get("name", ""),
                "position": p.get("position", player_info.get("position", "")),
                "shirt_number": p.get("shirtNumber", ""),
                "substitute": p.get("substitute", ""),
            }
            stats = p.get("statistics", {}) or {}
            for stat_key, stat_val in stats.items():
                row[stat_key] = stat_val
            rows.append(row)
    return rows


def scrape_matchweek(matchweek, season_year_label="26/27"):
    mw_str = f"mw{matchweek:02d}"
    out_dir = os.path.join(RAW_DIR, mw_str)
    raw_json_dir = os.path.join(out_dir, "raw_json")
    os.makedirs(raw_json_dir, exist_ok=True)

    print(f"Looking up season id for {season_year_label}...")
    season_id = get_season_id(season_year_label)
    print(f"Season id: {season_id}. Fetching matchweek {matchweek} fixtures...")

    events = get_events_for_round(season_id, matchweek)
    if not events:
        print(f"No events found for matchweek {matchweek} - has it been scheduled yet on Sofascore?")
        return

    team_rows = []
    player_rows = []

    for event in events:
        home = event.get("homeTeam", {}).get("name", "UNKNOWN_HOME")
        away = event.get("awayTeam", {}).get("name", "UNKNOWN_AWAY")
        event_id = event["id"]
        status = event.get("status", {}).get("type", "unknown")

        if status != "finished":
            print(f"  Skipping {home} vs {away} (status: {status}, not finished yet)")
            continue

        print(f"  Scraping {home} vs {away} (event {event_id})...")

        stats = get_event_statistics(event_id)
        time.sleep(REQUEST_DELAY_SECONDS)
        with open(os.path.join(raw_json_dir, f"{event_id}_statistics.json"), "w") as f:
            json.dump(stats, f, indent=2)

        home_row = {"team": home, "opponent": away, "venue": "home", "event_id": event_id}
        away_row = {"team": away, "opponent": home, "venue": "away", "event_id": event_id}
        for stat_key, vals in stats.items():
            home_row[stat_key] = vals.get("home")
            away_row[stat_key] = vals.get("away")
        team_rows.extend([home_row, away_row])

        lineups = get_event_lineups(event_id)
        time.sleep(REQUEST_DELAY_SECONDS)
        with open(os.path.join(raw_json_dir, f"{event_id}_lineups.json"), "w") as f:
            json.dump(lineups, f, indent=2)

        for p_row in lineups:
            team_name = home if p_row["side"] == "home" else away
            opp_name = away if p_row["side"] == "home" else home
            full_row = {
                "player_name": p_row.get("player_name", ""),
                "team": team_name,
                "opponent": opp_name,
                "venue": p_row["side"],
                "event_id": event_id,
            }
            full_row.update({k: v for k, v in p_row.items() if k not in ("side", "player_name")})
            player_rows.append(full_row)

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
        print("No player rows written - check raw_json/*_lineups.json to see what came back.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape a PL matchweek from Sofascore.")
    parser.add_argument("--matchweek", type=int, required=True, help="Matchweek number (1-38)")
    parser.add_argument("--season-label", default="26/27", help="Sofascore season year label, e.g. '26/27'")
    args = parser.parse_args()
    scrape_matchweek(args.matchweek, args.season_label)
