# Football agents for FPL

Agents and statistical models that pick Fantasy Premier League transfers, captain and starting XI.

By Nik Hannigan and Alec Diaz · *README last updated 2026-11-12 22:45*

## What it does

- **Friday:** refreshes the data, finds the best transfers (or "hold") for your team, checks team news for the players that matter, and writes a report.
- **Tuesday:** reviews the gameweek just played: how your team did, and whether our model or FPL's own `ep_next` predicted it better.

Code computes every number. An LLM is used only to read team news and write a short summary paragraph.

## Setup

Windows, Python 3.14.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install requests pandas pyarrow numpy scipy "pulp<4" highspy nodriver
```

Create `.env` in the repo root (gitignored):

```
OPENROUTER_API_KEY=...
# NEWS_MODEL=anthropic/claude-haiku-4.5   (optional override)
```

`nodriver` and Chrome are only needed for the FotMob scraper.

## Weekly use

```bash
python src/friday.py --team 123456     # before the deadline
python src/tuesday.py                  # after the gameweek finishes
```

Useful flags: `--skip-refresh` (reuse data on disk), `--no-summary` (no LLM), `--verbose`. Other options (`--horizon`, `--ft`, `--bank`, `--margin`) pass through to the optimiser. Without `--team`, the team used last is used.

## How it works

| Step | Script | What it does |
|---|---|---|
| 1. Scrape | `src/scrapers/fpl.py`, `fpl_fixtures.py`, `fpl_history.py` | Official FPL API: players, fixtures, per-gameweek history |
| | `src/scrapers/fotmob_scraper.py` | Extra match stats from FotMob (optional) |
| | `src/scrapers/fpl_last_season.py` | 2025-26 data from the vaastav archive, for backtests |
| 2. Ingest | `src/ingest.py` | One table, a row per player per gameweek; FPL ID is the master key |
| 3. Model | `src/models/baseline.py` | Expected points = P(appears) x points per appearance (shrunk to position average) |
| | `src/models/team_strength.py` | Poisson attack/defence ratings: team xG and clean-sheet chance per fixture |
| 4. Optimise | `src/optimise/transfers.py` | Integer programme (PuLP): transfers, XI, captain over the next 2 GWs |
| 5. News | `src/agents/news_agent.py` | Web-search LLM turns injury news into chance of playing |
| | `src/agents/runner.py` | Optimise → check news on the players that matter → optimise again |
| 6. Log | `src/predict_next.py` | Saves this week's predictions next to FPL's `ep_next` |
| 7. Evaluate | `src/evaluate/` | Walk-forward backtests of players, teams and transfers; model vs `ep_next` |

`src/friday.py` and `src/tuesday.py` run these in order; `src/pipeline.py` holds shared helpers.

**Default points model (m5):** m3 x a Poisson team-model factor per fixture.
**Optimiser defaults:** horizon 2 GWs, week weight 0.9, noise margin 2 pts per transfer, value 1 per banked free transfer (counted up to 2). Chips are ignored.

## Results so far

- Player model: rank correlation with actual points about 0.52-0.57. Most of the signal is "will he play" plus recent form. The fixture adjustment helps only slightly.
- Transfer replay of 2025-26: transfers beat keeping the GW4 squad (about 1,930 vs 1,770 points), but one season can't separate settings that differ by under ~100 points.

Details are in the project docs on claude.ai (backtest results, optimiser design, column guide).

## Data and outputs

Everything under `data/` is gitignored (except `data/processed/id_map.csv`). Saved outputs are timestamped:

- `data/predictions/` weekly predictions
- `data/recommendations/<team_id>_<name>/` optimiser output per team
- `data/news/` news agent runs
- `data/runs/` runner logs
- `data/reports/` Friday and Tuesday reports, plus `ledger.csv`
- `data/backtest/` backtest results

## Open work

- Player model based on share of team xG, so fixtures move expected points more.
- A captaincy "ceiling" term (the model currently rates some defenders highest).
- Decide whether FotMob/Sofascore stats earn their place in the backtest.
