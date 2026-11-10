# predict_next.py: column guide
 
Output of `python src/predict_next.py`, saved to `data/predictions/baseline_gw<N>_<date>.csv` / `.parquet`. One row per player with a fixture in the gameweek.
 
## Who and when
| Column | Meaning |
|---|---|
| `gameweek` | The gameweek being predicted. |
| `player_id` | FPL's player ID. The master key for joining to other tables. |
| `web_name` | The short name FPL shows (e.g. "Salah"). |
| `team` | The player's current club. |
| `position` | GKP, DEF, MID or FWD. |
| `price` | Current price in £m. |
 
## The fixture
| Column | Meaning |
|---|---|
| `opponent` | Opponent(s) that gameweek. A double gameweek shows "Chelsea + Hull City". |
| `was_home` | "H" or "A" for each fixture, in the same order as `opponent` (e.g. "H + A"). Despite the name, it is text. |
| `n_fixtures` | 1 normally, 2 in a double gameweek. Blank-week players are not in the table. |
 
## The model
| Column | Meaning |
|---|---|
| `p_appear` | Recency-weighted share of recent gameweeks in which the player got on the pitch (half-life 4 GWs). 1.0 = always played, 0 = no history. It is "gets minutes", not "starts". |
| `m3_availability_x_ppa` | Expected points = `p_appear` x points per appearance (shrunk towards the position average, so a one-goal cameo doesn't rank high). Summed over fixtures. The default ranking model. |
| `m4_with_fixture` | `m3` x opponent-weakness factor x home/away factor. Shown for comparison; it didn't beat `m3` in the backtest. |
| `avail` | FPL's chance of playing next round / 100. Blank in the API counts as 1.0 (fully fit). |
| `xpts` | The headline number: the chosen model (`--model`, default `m3`) x `avail`. Sorted on this. |
 
## FPL's own view
| Column | Meaning |
|---|---|
| `ep_next` | FPL's own expected points for the next gameweek, as of the last `fpl.py` run. Compare it with `xpts` after the gameweek is played. |
| `status` | a = available, d = doubtful, i = injured, s = suspended, u = unavailable, n = not available (e.g. left the club). |
| `news` | FPL's injury / availability text. |
 
## Reading it
- `xpts` is for ranking players against each other, not an exact forecast. Single gameweeks are very noisy.
- `ep_next` and the injury columns are only as fresh as the last `fpl.py` run. Run it first.
- Players with no appearances yet have `p_appear` = 0 and `xpts` = 0.