# Roblox Top Trending forecaster

An entry for make.fun's bounty **"Predict tomorrow's top 10 trending Roblox
games"** (`roblox-trending-forecast-01`).

`main.py` reads the history of Roblox's Top Trending chart (device `computer`,
country `us`) from `/input` and writes one forecast to
`/output/forecast.json`: for each of the top 10 places, a probability for every
game it names being at that place, plus a point guess at the top 10 in order.
That single file is scored against every hourly read of the 24 hours after
entries close.

Python 3.13, standard library only. No network, no model calls, no randomness,
no clock: the same history always writes the same bytes.

## Run it

```
python3 main.py
```

The paths default to `/input` and `/output`, as in the grader's container. To
run it anywhere else, point the two environment variables at your directories:

```
MAKEFUN_INPUT_DIR=./input MAKEFUN_OUTPUT_DIR=./out python3 main.py
```

`/input` holds:

| file | what it is |
| --- | --- |
| `history.ndjson` | one read of the chart per line, oldest first: `at`, `top_trending` (each game's `rank`, `universe_id`, `name`, `playing`, …), `top_playing_now`, `details` |
| `task.json` | the scored window (`window.from`, `window.until`, `slots`, `slot_minutes`), `places`, `output` |

The scored window is read from `task.json`, never hard-coded, because the
practice run and the final run have different windows.

### Try it on sample data

Any JSON-lines file of chart reads works. A minimal one, with two reads:

```bash
mkdir -p /tmp/demo/input
python3 - <<'PY'
import json
reads = []
for hour, order in enumerate([[1, 2, 3, 4, 5], [1, 3, 2, 4, 5]]):
    reads.append({"at": f"2026-11-09T{hour:02d}:00:00Z", "source": "demo",
                  "filters": None, "ok": True, "top_playing_now": None,
                  "details": None, "responses": {},
                  "top_trending": [{"rank": i + 1, "universe_id": uid,
                                    "root_place_id": uid, "name": f"Game {uid}",
                                    "playing": 1000 - 10 * i, "up_votes": None,
                                    "down_votes": None, "genre": None}
                                   for i, uid in enumerate(order)]})
with open("/tmp/demo/input/history.ndjson", "w") as fh:
    for read in reads:
        fh.write(json.dumps(read) + "\n")
json.dump({"history_until": "2026-11-09T01:00:00Z", "places": 10,
           "window": {"from": "2026-11-09T02:00:00Z", "until": "end",
                      "slots": 24, "slot_minutes": 60},
           "output": "forecast.json", "snapshots": len(reads)},
          open("/tmp/demo/input/task.json", "w"))
PY
MAKEFUN_INPUT_DIR=/tmp/demo/input MAKEFUN_OUTPUT_DIR=/tmp/demo/out python3 main.py
python3 -m json.tool /tmp/demo/out/forecast.json | head -20
```

The output is one object with exactly two keys:

```json
{"probabilities": [{"universe_id": 7529591378, "place": 1, "p": 0.61},
                   {"universe_id": 994732206,  "place": 1, "p": 0.24}],
 "top10": [7529591378, 994732206, ...]}
```

Each place sums to at most 1; whatever a place leaves unnamed is what the
scorer spreads over the games that read's chart has and the forecast did not
name at that place.

## How it forecasts

The chart is an *ordering*, and the ordering keeps moving, so the model works
in log-rank space: `log(place)`. That is the natural scale for a chart that
turns over by shifting games past each other — a slip from 1st to 2nd matters
much more than one from 41st to 42nd.

**1. A distribution per game, per hour.** Every game on the latest chart gets a
distribution over where it could be `h` hours later: its current log-rank,
shifted by a damped drift and spread by a normal whose width grows with
`sqrt(h)`.

*The spread* is scaled by how jumpy that particular game has been recently,
measured from consecutive reads as `|change of log-rank| / sqrt(hours)` and
shrunk toward the chart's own average when there are only a few pairs to go on.
A heavy-tailed second component catches sudden jumps at a small probability, so
they are not priced at zero. Measured on the archive, a game's moves turn out
to be close to a random walk in log-rank — which is why the width grows with
`sqrt(h)` and not with `h`.

*The drift* comes from two effects that survive a backtest:

- **Player growth.** The chart ranks the largest *relative increase in daily
  active users*, so a game whose concurrent players are growing faster than the
  rest of the chart's tends to climb. The comparison is against the same game
  about a day earlier, and with both sides divided by their read's median, so
  the daily cycle that lifts the whole chart at once cancels out.
- **Ageing.** The growth that put a game on the chart is measured over the past
  week, so a game that has been climbing for days has already banked most of it
  and drifts back down. A long unbroken stay on the chart is a reason to expect
  a fall, not a rise.

*Rank momentum* — carrying a game's recent climb or fall forward — sounds
obvious and is wrong here: mean reversion dominates it, and every setting of it
scored worse in the backtests, so it is off.

**2. Places are balanced against each other.** Ten games each given a 30%
chance of being first would claim three first places. Sinkhorn balancing
(a transportation problem) rescales the games-by-places matrix until every game
holds exactly one place and every place exactly one game, which is what an
ordered chart means. This step is worth about two nats a read.

**3. Breakouts are named, not just modelled.** A game can leap from far down the
chart into the top 10 in a day — in the practice window two games came in from
ranks 35 and 42, and one of them had multiplied its players eightfold a few
hours earlier. So the chart is scanned below the top 10 for games whose players
are multiplying, and the single strongest is named with a modest chance of
arriving at each place. Its mass is taken out of the places before the
balancing, so it is a real hedge rather than a relabelling.

**4. Two scenarios the diffusion cannot express** are mixed in, both growing
with the horizon:

- *frozen*: a chart is computed, not a live count, so between recomputes it does
  not move at all;
- *dropout*: a leader can leave outright, and every game below moves up one
  place.

**5. Hedging.** Each place names 98% of its probability explicitly and leaves
the rest for games it did not name. Under a log score a confident wrong call
costs far more than a hedged right one costs, so the file is deliberately a
little less sharp than the model's beliefs.

**6. One file for 24 hours.** The 24 hourly slots are forecast separately and
then averaged. Under a log score, when one forecast is scored against every
read of a window, the average of the per-slot forecasts is the best single
answer — betting on any one hour is strictly worse in expectation.

## How it was validated

Parameters were fitted by rolling-origin backtests on the public MIT-licensed
chart archive ([Kenny-Dhub/roblox-snapshots](https://github.com/Kenny-Dhub/roblox-snapshots),
41 reads from 2026-10-30 to 2026-11-09). Each read in turn ends a history, the
program forecasts the following 24 hours, and the result is scored with this
contest's log score against the archive's reads in that window — 35 cutoffs.

| | mean log score |
| --- | --- |
| house persistence (the chart does not move) | −39.24 |
| this model | **−18.20** |

On the two halves of the archive separately (early cutoffs / late cutoffs) the
model scores −17.8 and −18.5, and the parameters that were fitted on one half
transfer to the other, so the fit is not chasing one half's noise. The
practice window is the hardest in the archive — the #1 left the chart, two games
came in from rank 35 and 42 — and scores about −25, still far above the
baseline's −62 there.

Beyond the score, every run is checked against the contest's format rules: the
two-key shape, positive integer ids, places 1–10, `p` in [0, 1], no game named
twice at a place, at most 100 rows a place, each place summing to at most 1,
ten distinct ids in `top10`, under 1 MiB, and byte-identical on a second run.
Runs take about a second.

The program is also robust to the history being damaged — broken lines, reads
flagged not-ok, missing player counts, missing ranks, a chart of three games,
duplicate timestamps, string-typed numbers, or no history at all. It skips what
it cannot use, and if the model itself ever failed it falls back to the house
persistence baseline, so a run always writes a valid forecast.

## Licence

MIT, see [LICENSE](LICENSE).
