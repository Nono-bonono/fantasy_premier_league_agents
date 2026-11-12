"""
news_agent.py - team-news agent for the FPL project.

One web-search LLM call per club (via OpenRouter) turns press conferences and injury news into
structured availability: for each player it finds news on, a chance of playing (0-100), a short
evidence note, a source URL + date and a confidence. Nothing else - it does not pick transfers.

How it plugs in (not wired up yet - see apply_news):
    news  = load_latest(gw)                       # newest data/news/news_gw<N>_<timestamp>.csv
    avail = apply_news(avail, news)               # replaces FPL's chance_of_playing where news is better

Safeguards:
  * the model may only report players from the list we give it, by FPL player_id (no name matching)
  * a report counts as "verified" only if its source_url was among the pages the search tool returned
  * apply_news ignores unverified reports, ignores low confidence, and only RAISES a player's
    availability above FPL's flag on high confidence
  * every run is saved with a timestamp (csv for the table, json for raw responses and cost), so
    you can backtest the agent later

USAGE:
    python src/agents/news_agent.py                      # next gameweek, all clubs
    python src/agents/news_agent.py --teams Arsenal,Chelsea --max-results 3
    python src/agents/news_agent.py --model moonshotai/kimi-k2.6
    python src/agents/news_agent.py --dry-run            # print one club's prompt, no API call

NEEDS: OPENROUTER_API_KEY in the environment or in .env at the repo root.
INSTALL: pip install requests pandas pyarrow
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
NEWS_DIR = os.path.join(BASE_DIR, "data", "news")
GW_PATH = os.path.join(PROC_DIR, "player_gw.parquet")
PLAYERS_PATH = os.path.join(PROC_DIR, "fpl_players.parquet")
FIXTURES_PATH = os.path.join(PROC_DIR, "fpl_team_fixtures.parquet")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
CONF_RANK = {"low": 0, "medium": 1, "high": 2}
MAX_PLAYERS_PER_TEAM = 30
MIN_MINUTES = 90          # players with fewer minutes are only listed if FPL flags them


def load_env():
    """Tiny .env reader (KEY=VALUE lines) so we don't need python-dotenv."""
    path = os.path.join(BASE_DIR, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


load_env()
DEFAULT_MODEL = os.environ.get("NEWS_MODEL", "anthropic/claude-haiku-4.5")


# ---------------------------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------------------------
SYSTEM = """You check Premier League team news for a Fantasy Premier League tool. Use web search.

Rules:
- Use only sources published in the last {days} days. Prefer the manager's press conference, the club's own site, BBC Sport, the Premier League site and major newspapers.
- Report only players from the list in the user message, by their player_id. Ignore everyone else.
- Report a player only when a source gives NEW or CLEARER information than the FPL flag shown: injury, return from injury, suspension, illness, or a clear statement that he is fit. Tactical rotation or "may be rested" is not availability news - leave it out.
- Never guess. If you find nothing, return {{"players": []}}.
- Write the evidence in your own words, at most 200 characters.

chance_of_playing: 0 = ruled out, 25 = major doubt, 50 = 50/50, 75 = minor doubt, 100 = fit / expected to be available.
confidence: high = manager or club says it directly within the last 3 days; medium = credible report; low = rumour or older than a week.

Reply with ONLY this JSON, no other text:
{{"players": [{{"player_id": 123, "chance_of_playing": 75, "evidence": "...", "source_url": "https://...", "source_date": "YYYY-MM-DD", "confidence": "high"}}]}}"""


def build_messages(team, fx, cands, days):
    lines = ["player_id | name | pos | FPL status | FPL chance | FPL news"]
    for r in cands.itertuples():
        chance = "" if pd.isna(r.chance_of_playing_next_round) else int(r.chance_of_playing_next_round)
        news = (r.news or "").replace("\n", " ")[:120]
        lines.append(f"{r.id} | {r.web_name} | {r.position} | {r.status} | {chance} | {news}")
    where = "home" if fx["was_home"] else "away"
    user = (f"Today is {date.today().isoformat()}.\n"
            f"Team: {team}. Gameweek {fx['gameweek']}: {where} vs {fx['opponent']}, kickoff {fx['kickoff_time']}.\n"
            f"Find the latest availability news for {team} ahead of this match.\n\n"
            f"Players you may report on:\n" + "\n".join(lines))
    return [{"role": "system", "content": SYSTEM.format(days=days)}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------------------------
# Model call and parsing
# ---------------------------------------------------------------------------------------------
def call_model(body, key, timeout=120, retries=2):
    """POST to OpenRouter; retry on rate limits and server errors (not on timeouts). Kept separate so tests can swap it."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "X-Title": "fpl-agents news agent"}
    for attempt in range(retries + 1):
        r = requests.post(API_URL, headers=headers, json=body, timeout=timeout)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(3 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()


def norm_url(u):
    return re.sub(r"[#?].*$", "", (u or "").strip()).rstrip("/").lower()


def cited_urls(message):
    urls = set()
    for a in message.get("annotations") or []:
        u = (a.get("url_citation") or {}).get("url") or a.get("url")
        if u:
            urls.add(norm_url(u))
    return urls


def parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < 0:
        raise ValueError("no JSON object in reply")
    return json.loads(text[i:j + 1])


def validate(items, allowed_ids, cited):
    """Keep well-formed reports for players we asked about. Returns (rows, problems)."""
    rows, problems = [], []
    for it in items:
        try:
            pid = int(it["player_id"])
            chance = int(round(float(it["chance_of_playing"])))
            conf = str(it.get("confidence", "")).lower()
            url = str(it.get("source_url") or "").strip()
            if pid not in allowed_ids:
                raise ValueError(f"player_id {pid} not in the list we sent")
            if not 0 <= chance <= 100:
                raise ValueError(f"chance {chance} out of range")
            if conf not in CONF_RANK:
                raise ValueError(f"bad confidence {conf!r}")
            if not url:
                raise ValueError("no source_url")
            rows.append({"player_id": pid, "chance_of_playing": chance, "confidence": conf,
                         "evidence": str(it.get("evidence", ""))[:300], "source_url": url,
                         "source_date": str(it.get("source_date") or ""),
                         "verified": norm_url(url) in cited})
        except (KeyError, TypeError, ValueError) as e:
            problems.append(f"{e}: {it}")
    return rows, problems


def run_team(team, fx, cands, args, key):
    """One club. Never raises - returns a record with ok/error so one failure doesn't stop the run."""
    rec = {"team": team, "ok": False, "n_candidates": len(cands)}
    t0 = time.time()
    body = {"model": args.model, "temperature": 0, "max_tokens": 2000,
            "messages": build_messages(team, fx, cands, args.days),
            "tools": [{"type": "openrouter:web_search",
                       "parameters": {"max_results": args.max_results, "max_total_results": args.max_results * 2}}]}
    if not args.model.startswith("anthropic/"):
        body["reasoning"] = {"effort": "low"}      # reasoning models (e.g. Kimi) can think for minutes otherwise
    try:
        resp = call_model(body, key, timeout=args.timeout)
        msg = resp["choices"][0]["message"]
        cited = cited_urls(msg)
        data = parse_json(msg.get("content"))
        rows, problems = validate(data.get("players", []), set(cands["id"].astype(int)), cited)
        rec.update(ok=True, rows=rows, problems=problems, cited=sorted(cited),
                   usage=resp.get("usage"), raw=msg.get("content"))
    except Exception as e:  # noqa: BLE001 - we want to log any failure and carry on
        rec.update(error=f"{type(e).__name__}: {e}", rows=[])
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


# ---------------------------------------------------------------------------------------------
# Using the output
# ---------------------------------------------------------------------------------------------
def load_latest(gw, news_dir=NEWS_DIR, max_age_hours=36):
    """
    All news tables for a gameweek saved in the last max_age_hours, stacked (the runner checks players in
    several rounds, each saved as its own file). Older runs are ignored: stale news shouldn't beat a fresh
    FPL flag. Adds a 'run' column (the file's timestamp). None if there is nothing recent.
    """
    if not os.path.isdir(news_dir):
        return None
    parts = []
    for f in sorted(os.listdir(news_dir)):
        m = re.fullmatch(rf"news_gw{gw}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{4}})\.csv", f)
        if not m:
            continue
        if (datetime.now() - datetime.strptime(m.group(1), "%Y-%m-%d_%H%M")).total_seconds() > max_age_hours * 3600:
            continue
        d = pd.read_csv(os.path.join(news_dir, f))
        d["run"] = m.group(1)
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else None


def apply_news(avail, news, min_confidence="medium", require_verified=True):
    """
    avail : DataFrame with player_id, avail (0-1, from FPL's flag).
    news  : table from load_latest / this script's csv.
    Returns a copy with avail updated where news is usable, plus avail_source ('fpl' or 'news').
    Rules: verified source only; confidence >= min_confidence; to RAISE avail above FPL's flag the
    confidence must be 'high'. If a player has several reports, the newest source_date wins.
    """
    out = avail.copy()
    out["avail_source"] = "fpl"
    if news is None or len(news) == 0:
        return out
    n = news[news["confidence"].map(CONF_RANK) >= CONF_RANK[min_confidence]].copy()
    if require_verified:
        n = n[n["verified"].astype(bool)]
    n["source_date"] = n["source_date"].fillna("").astype(str)
    n = n.sort_values(["source_date"] + (["run"] if "run" in n.columns else [])).drop_duplicates("player_id", keep="last").set_index("player_id")
    new = out["player_id"].map(n["chance_of_playing"]) / 100
    conf = out["player_id"].map(n["confidence"])
    use = new.notna() & ((new <= out["avail"]) | (conf == "high"))
    out.loc[use, "avail"] = new[use]
    out.loc[use, "avail_source"] = "news"
    return out


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Search the web for team news and save structured availability.")
    ap.add_argument("--gw", type=int, help="gameweek (default: last played + 1)")
    ap.add_argument("--teams", help="comma-separated club names, as in fpl_players.parquet (default: all)")
    ap.add_argument("--players", help="comma-separated FPL player_ids: check only these players (and only their clubs)")
    ap.add_argument("--watchlist", help="a recommendation .json from transfers.py: check only the players on its watchlist")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model id")
    ap.add_argument("--max-results", type=int, default=3, help="search results per club (Exa is billed per result)")
    ap.add_argument("--days", type=int, default=7, help="ignore sources older than this")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120, help="seconds to wait for one club before giving up on it")
    ap.add_argument("--dry-run", action="store_true", help="print the first club's prompt and exit")
    ap.add_argument("--out-dir", default=NEWS_DIR)
    args = ap.parse_args(argv)

    for path, hint in ((GW_PATH, "ingest.py"), (PLAYERS_PATH, "fpl.py"), (FIXTURES_PATH, "fpl_fixtures.py")):
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path} - run {hint} first.")
    players = pd.read_parquet(PLAYERS_PATH)
    tf = pd.read_parquet(FIXTURES_PATH)
    gw = args.gw or int(pd.read_parquet(GW_PATH)["gameweek"].max()) + 1

    snap = datetime.fromtimestamp(os.path.getmtime(PLAYERS_PATH)).date()
    if snap != date.today():
        print(f"[warn] fpl_players.parquet is from {snap}, not today - the FPL flags shown to the model may be stale. Run fpl.py first.")

    fx_gw = tf[tf["gameweek"] == gw].sort_values("kickoff_time").groupby("team").first().reset_index()
    teams = sorted(fx_gw["team"])
    if args.teams:
        want = {t.strip().lower() for t in args.teams.split(",")}
        teams = [t for t in teams if t.lower() in want]
        if not teams:
            raise SystemExit(f"No matching clubs with a GW{gw} fixture. Check spelling against fpl_players.parquet.")
    fx_by_team = fx_gw.set_index("team").to_dict("index")
    for t, f in fx_by_team.items():
        f["gameweek"] = gw

    only = set()
    if args.players:
        only |= {int(x) for x in args.players.split(",") if x.strip()}
    if args.watchlist:
        with open(args.watchlist, encoding="utf-8") as f:
            only |= {int(w["player_id"]) for w in json.load(f).get("watchlist", [])}

    cands = {}
    for t in teams:
        p = players[players["team"] == t]
        if args.players or args.watchlist:
            keep = p["id"].isin(only)
        else:
            keep = (p["minutes"].fillna(0) >= MIN_MINUTES) | (p["status"] != "a") | (p["news"].fillna("") != "")
        keep &= ~p["status"].isin(["u", "n"])      # left the club / permanently unavailable: nothing to find
        cands[t] = p[keep].sort_values("minutes", ascending=False).head(MAX_PLAYERS_PER_TEAM)
    teams = [t for t in teams if len(cands[t])]
    if not teams:
        raise SystemExit("No players to check (empty selection, or none of them have a fixture this gameweek).")

    if args.dry_run:
        t = teams[0]
        for m in build_messages(t, fx_by_team[t], cands[t], args.days):
            print(f"--- {m['role']} ---\n{m['content']}\n")
        print(f"[dry run] {len(teams)} club(s) would be searched with {args.model}, "
              f"{args.max_results} search results each.")
        return

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set (environment or .env at the repo root).")

    n_players = sum(len(cands[t]) for t in teams)
    print(f"GW{gw}: searching news for {n_players} players at {len(teams)} clubs with {args.model} ...")
    recs = []
    ex = ThreadPoolExecutor(max_workers=args.workers)
    futs = {ex.submit(run_team, t, fx_by_team[t], cands[t], args, key): t for t in teams}
    interrupted = False
    try:
        for f in as_completed(futs):
            r = f.result()
            recs.append(r)
            status = f"{len(r['rows'])} report(s)" if r["ok"] else "FAILED " + r["error"][:80]
            print(f"  [{len(recs)}/{len(teams)}] {r['team']}: {status} ({r['seconds']}s)", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print(f"\nInterrupted - saving the {len(recs)} club(s) finished so far.", flush=True)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    recs.sort(key=lambda r: r["team"])

    info = players.set_index("id")
    table = []
    for r in recs:
        for row in r["rows"]:
            p = info.loc[row["player_id"]]
            table.append({"gameweek": gw, "team": r["team"], "web_name": p["web_name"],
                          "fpl_status": p["status"], "fpl_chance": p["chance_of_playing_next_round"],
                          **row, "model": args.model})
    cols = ["gameweek", "team", "player_id", "web_name", "fpl_status", "fpl_chance", "chance_of_playing",
            "confidence", "verified", "evidence", "source_url", "source_date", "model"]
    df = pd.DataFrame(table, columns=cols)

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    stem = os.path.join(args.out_dir, f"news_gw{gw}_{stamp}")
    df.to_csv(stem + ".csv", index=False, encoding="utf-8")
    meta = {"gameweek": gw, "timestamp": stamp, "model": args.model, "max_results": args.max_results,
            "days": args.days, "teams": recs}
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=str)

    failed = [r for r in recs if not r["ok"]]
    cost = sum((r.get("usage") or {}).get("cost") or 0 for r in recs)
    print(f"\n{len(df)} player report(s) from {len(recs) - len(failed)}/{len(recs)} clubs; "
          f"{int(df['verified'].sum()) if len(df) else 0} with a verified source. "
          f"Reported cost ${cost:.3f}." if cost else
          f"\n{len(df)} player report(s) from {len(recs) - len(failed)}/{len(recs)} clubs.")
    for r in failed:
        print(f"  [failed] {r['team']}: {r['error']}")
    for r in recs:
        for pr in r.get("problems", []):
            print(f"  [dropped] {r['team']}: {pr}")
    if len(df):
        pd.set_option("display.width", 220)
        pd.set_option("display.max_colwidth", 60)
        print(df[["team", "web_name", "fpl_status", "fpl_chance", "chance_of_playing", "confidence",
                  "verified", "source_date", "evidence"]].to_string(index=False))
    print(f"\nSaved to {stem}.csv / .json")
    if interrupted:
        print(f"Not searched: {sorted(set(teams) - {r['team'] for r in recs})}", flush=True)
        os._exit(0)     # calls still in flight would otherwise keep the process alive for up to --timeout seconds
    return {"table": df, "stem": stem, "cost": cost, "clubs": len(recs), "failed": len(failed)}


if __name__ == "__main__":
    main()
