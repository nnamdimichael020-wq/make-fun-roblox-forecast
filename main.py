#!/usr/bin/env python3
"""Forecast Roblox's Top Trending chart place by place for the next 24 hours.

Entry for make.fun's ``roblox-trending-forecast-01``.

Input  (``$MAKEFUN_INPUT_DIR``, else ``/input``)
    ``history.ndjson``: one read of the chart per line, oldest first.
    ``task.json``: the scored window (``window.from``, ``slots``,
    ``slot_minutes``), the number of places, and the output file name.
Output (``$MAKEFUN_OUTPUT_DIR``, else ``/output``)/forecast.json
    For each of the top 10 places, a probability for every game named at that
    place, and a point guess at the top 10 in order.

How it forecasts
----------------
The chart is an ordering of games, and the ordering keeps moving.  So the
model works in log-rank space: each game's distance above or below its
neighbours is ``log(place)``, which is the natural scale on which a chart like
this turns over -- a slip from 1st to 2nd is a bigger event than 41st to 42nd.

1. Every game on the latest chart gets a distribution over where it could be
   ``h`` hours later: its current log-rank, shifted by a damped drift and
   spread by a normal whose width grows with ``sqrt(h)``, widened by how jumpy
   that game has been and carrying a heavy-tailed component so sudden jumps are
   not priced at zero.  The drift comes from two measured effects: player
   growth against a day earlier (the chart ranks *growth in daily players*, so
   a game out-growing the field tends to climb) and the ageing of a long stay
   on the chart (the growth has been banked, so it fades).
2. Places are balanced against each other (Sinkhorn): every place ends up
   holding exactly one game's worth of probability, which is what an ordered
   chart means.  Without this step, ten games with a 30% chance of being first
   would claim three first places.
3. Each place also keeps some probability unnamed, for games that are not on
   the last chart at all -- one can arrive at any time, and a place that named
   nothing would spread its whole probability over the chart.
4. The 24 hourly slots are forecast separately and averaged into the one file
   the harness scores against every read: under a log score that average is the
   best single forecast for the window.

Standard library only, no network, no model calls.  No randomness and no
clock: the same history always writes the same bytes.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Calibration: log-rank units (natural log of the place) and hours.
#
# Fitted by rolling-origin backtests over the public chart archive: each read
# of the archive ends a history, the model forecasts the following 24 hours,
# and the forecast is scored with this contest's log score.  The handful of
# parameters that survived an out-of-sample check are listed in README.md.
# ---------------------------------------------------------------------------
PARAMS = {
    # --- spread of the log-rank over the horizon --------------------------
    "sigma0": 0.05,           # log-rank spread at horizon 0
    "sigma_hour": 0.075,      # log-rank spread per sqrt(hour) at the top of the chart
    "sigma_place_mult": 0.0,  # extra spread further down: * (1 + mult * log(place)/log(places))
    "jitter_share": 1.0,     # how much of the width is the game's own jumpiness
    "jitter_min": 0.75,
    "jitter_max": 1.4,
    "tail_weight": 0.05,      # weight of the wide component (sudden jumps)
    "tail_mult": 2.0,         # its width, as a multiple of the narrow one

    # --- drift from player growth -----------------------------------------
    "surge_lag": 22.0,        # hours: players now against players this long ago
    "surge_share": 0.35,      # log-rank climb per unit of relative player growth
    "surge_tau": 12.0,        # hours; the climb builds up with this constant
    "surge_cap": 1.5,         # largest player surge counted, in log units
    "surge_from": 0.35,       # minimum surge at which a game is treated as growing

    # --- ageing: the chart ranks a *week's* growth, so a long stay ages out ---
    "tenure_drift": 0.04,     # extra log-rank fall per day of stay above the median
    "tenure_width": 0.0,     # extra width per day of stay above the median

    # --- leaving and entering the chart ------------------------------------
    "entry_rate": 0.006,     # per-hour mass each place gives to unseen games
    "entry_cap": 0.08,        # most of a place's mass that may go to unseen games
    "entry_slope": 2.0,      # lower places take proportionally more
    "entrant_depth": 60,     # how far below the top 10 a named entrant may sit
    "entrant_surge": 1.2,     # relative player growth that marks a game out as arriving
    "entrant_boost": 0.35,    # extra arrival chance per unit of growth beyond that
    "entrant_center": 4.5,    # the place a named entrant tends to land at
    "entrant_width": 0.7,    # log-place spread of where it lands
    "max_entrants": 1,        # at most this many named entrants at each place

    # --- assembling the file ----------------------------------------------
    "freeze_weight": 0.08,    # share of probability on "the chart does not move at all"
    "dropout_weight": 0.1,   # share on "the leader leaves and every game below moves up one"
    "unnamed_share": 0.98,    # share of each place named explicitly; the rest hedges
    "min_named_p": 1e-5,      # do not spend a row on less than this
    "max_rows_per_place": 70,  # the scorer allows 100
    "max_games": 80,          # games carried through the matrix
}


# ---------------------------------------------------------------------------
# Reading the input
# ---------------------------------------------------------------------------
def parse_time(value) -> float | None:
    """An ISO-8601 UTC stamp as POSIX seconds, or None if it cannot be read."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text[-1] in "Zz":
        text = text[:-1] + "+00:00"
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def as_number(value) -> float | None:
    """A finite float, or None.  Booleans and non-numbers are None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_reads(path: str) -> list[dict]:
    """Every usable read of the chart, oldest first.

    A read is ``{"t", "order", "playing"}``: when it started, its games in chart
    order, and each game's player count.  Broken lines, reads flagged not-ok
    and reads without a chart are skipped rather than fatal, so a slightly
    damaged history still gets a forecast.
    """
    reads: list[dict] = []
    try:
        handle = open(path, encoding="utf-8", errors="replace")
    except OSError:
        return reads
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict) or record.get("ok") is False:
                continue
            when = parse_time(record.get("at"))
            rows = record.get("top_trending")
            if when is None or not isinstance(rows, list):
                continue

            placed: list[tuple[float, int, int]] = []
            playing: dict[int, float] = {}
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                uid = row.get("universe_id")
                if not isinstance(uid, int) or isinstance(uid, bool) or uid < 1:
                    continue
                rank = as_number(row.get("rank"))
                placed.append((rank if rank is not None else float(index + 1), index, uid))
                count = as_number(row.get("playing"))
                if count is not None and count > 0:
                    playing[uid] = count
            placed.sort()

            order: list[int] = []
            seen: set[int] = set()
            for _, _, uid in placed:
                if uid not in seen:      # a game listed twice keeps its best place
                    seen.add(uid)
                    order.append(uid)
            if order:
                reads.append({"t": when, "order": order, "playing": playing})

    reads.sort(key=lambda read: read["t"])

    # Two reads at the same instant (a retried fetch): keep the fuller one.
    unique: list[dict] = []
    for read in reads:
        if unique and abs(read["t"] - unique[-1]["t"]) < 1.0:
            if len(read["order"]) > len(unique[-1]["order"]):
                unique[-1] = read
            continue
        unique.append(read)
    return unique


def load_task(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            task = json.load(handle)
    except (OSError, ValueError):
        return {}
    return task if isinstance(task, dict) else {}


def window_hours(task: dict, reads: list[dict], default_slots: int = 24) -> list[float]:
    """The horizon of each scored slot's read, in hours after the last read.

    The window is read from ``task.json``, never hard-coded: the practice run
    and the final run have different windows.
    """
    window = task.get("window") if isinstance(task.get("window"), dict) else {}
    slots = window.get("slots")
    if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
        slots = default_slots
    slots = min(slots, 240)
    minutes = as_number(window.get("slot_minutes"))
    step = (minutes / 60.0) if minutes and minutes > 0 else 1.0
    start = parse_time(window.get("from"))
    if start is None or not reads:
        return [step * (index + 1) for index in range(slots)]
    last = reads[-1]["t"]
    # A slot's read is the first read that started inside it, so mid-slot is
    # the fair guess at which hour offset the harness will score.
    return [max(0.25, (start - last) / 3600.0 + step * (index + 0.5))
            for index in range(slots)]


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------
SQRT2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    if x >= 0.0:
        return 1.0 - 0.5 * math.erfc(x / SQRT2)
    return 0.5 * math.erfc(-x / SQRT2)


def median(values) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def log_rank_bounds(places: int) -> list[float]:
    """Place k covers log(k - 1/2) .. log(k + 1/2), so it is centred on log(k)."""
    return [math.log(0.5)] + [math.log(k + 0.5) for k in range(1, places + 1)]


def log_ranks(read: dict) -> dict[int, float]:
    return {uid: math.log(index + 1) for index, uid in enumerate(read["order"])}


# ---------------------------------------------------------------------------
# What the history says about each game
# ---------------------------------------------------------------------------
class History:
    """The history, distilled into the per-game numbers the model needs."""

    def __init__(self, reads: list[dict], params: dict):
        self.reads = reads
        self.params = params
        self.last = reads[-1]
        self.order = list(self.last["order"])
        self.ranks = log_ranks(self.last)
        self.place = {uid: index + 1 for index, uid in enumerate(self.order)}
        self.surge = self._surges()
        self.jitter = self._jitters()
        self.surge_fast = self._fast_surges()
        self.tenure = self._tenures()
        measured = sorted(self.tenure.values())
        self.median_tenure = median(measured) if measured else 0.0

    # -- player growth -----------------------------------------------------
    def _surges(self) -> dict[int, float]:
        out: dict[int, float] = {uid: 0.0 for uid in self.order}
        reference = self._read_near(self.params["surge_lag"])
        if reference is None:
            return out
        now, then = self.last["playing"], reference["playing"]
        if len(now) < 3 or len(then) < 3:
            return out
        now_median, then_median = median(list(now.values())), median(list(then.values()))
        if now_median <= 0 or then_median <= 0:
            return out
        raw: dict[int, float] = {}
        for uid in self.order:
            after, before = now.get(uid), then.get(uid)
            if after is not None and before is not None:
                # Both sides relative to the chart's median, which takes out
                # the daily cycle that lifts or lowers every game at once.
                raw[uid] = math.log(after / now_median) - math.log(before / then_median)
            elif after is not None:
                raw[uid] = 0.8    # new to the chart since then: growing, unknown how fast
            else:
                raw[uid] = -0.8   # was here, has faded
        centre = median(list(raw.values()))
        cap = self.params["surge_cap"]
        for uid, value in raw.items():
            out[uid] = min(max(value - centre, -cap), cap)
        return out

    def _fast_surges(self) -> dict[int, float]:
        """Player growth over the last read or two: how fast a game is taking off.

        The daily comparison is the right one for a game's slow climb, but a
        breakout -- the kind that lifts a game from far down the chart straight
        into the top 10 -- shows in a few hours of player counts first.
        """
        out: dict[int, float] = {uid: 0.0 for uid in self.order}
        if len(self.reads) < 2:
            return out
        previous = self.reads[-2]
        now, then = self.last["playing"], previous["playing"]
        gap = (self.last["t"] - previous["t"]) / 3600.0
        if gap <= 0 or len(now) < 3 or len(then) < 3:
            return out
        now_median = median(list(now.values()))
        then_median = median(list(then.values()))
        if now_median <= 0 or then_median <= 0:
            return out
        # Normalise the growth by the gap, and by the chart's own move over it.
        for uid in self.order:
            after, before = now.get(uid), then.get(uid)
            if after is not None and before is not None and before > 0:
                growth = math.log(after / now_median) - math.log(before / then_median)
                out[uid] = growth * min(1.0, 6.0 / max(gap, 1.0))
        return out

    def _read_near(self, hours_ago: float) -> dict | None:
        """The read closest to ``hours_ago`` before the last one, if close enough."""
        if len(self.reads) < 2:
            return None
        target = self.last["t"] - hours_ago * 3600.0
        best, best_gap = None, None
        for read in self.reads[:-1]:
            gap = abs(read["t"] - target)
            if best_gap is None or gap < best_gap:
                best, best_gap = read, gap
        if best is None or best_gap is None or best_gap > 0.45 * hours_ago * 3600.0:
            return None
        return best

    # -- tenure ------------------------------------------------------------
    def _tenures(self) -> dict[int, float]:
        """Hours each game has been on the chart without a break, to the last read.

        The chart ranks the largest *relative increase in daily active users
        over the past week*: a game that has been climbing all week has already
        booked that growth, so a long stay is a reason to expect a fall.
        """
        out: dict[int, float] = {}
        for uid in self.order:
            hours = 0.0
            for earlier, later in zip(reversed(self.reads[:-1]), reversed(self.reads)):
                gap = (later["t"] - earlier["t"]) / 3600.0
                if uid not in earlier["order"]:
                    break
                hours += gap
            out[uid] = hours
        return out

    # -- jumpiness ---------------------------------------------------------
    def _jitters(self) -> dict[int, float]:
        """Each game's own log-rank volatility, relative to the chart's average."""
        floor = max(self._chart_volatility(), 0.05)
        params = self.params
        out: dict[int, float] = {}
        for uid in self.order:
            ratio = self._game_volatility(uid, floor) / floor
            blended = 1.0 + params["jitter_share"] * (ratio - 1.0)
            out[uid] = min(max(blended, params["jitter_min"]), params["jitter_max"])
        return out

    def _pairs(self, window: float = 72.0):
        """Consecutive reads a plausible gap apart, from the recent past."""
        last_t = self.last["t"]
        for earlier, later in zip(self.reads, self.reads[1:]):
            gap = (later["t"] - earlier["t"]) / 3600.0
            if gap < 0.5 or gap > 20.0:
                continue
            if (last_t - earlier["t"]) / 3600.0 > window:
                continue
            yield gap, log_ranks(earlier), log_ranks(later), math.log(len(later["order"]) + 1.0)

    def _chart_volatility(self) -> float:
        """Mean |change of log-rank| per sqrt(hour) over the top of the chart."""
        total, count = 0.0, 0
        for gap, before, after, edge in self._pairs():
            scale = math.sqrt(gap)
            for uid, start in before.items():
                if start > math.log(12.0):
                    continue
                total += abs(after.get(uid, edge) - start) / scale
                count += 1
        return total / count if count else 0.45

    def _game_volatility(self, uid: int, floor: float) -> float:
        total, pairs = 0.0, 0
        for gap, before, after, _ in self._pairs():
            start, end = before.get(uid), after.get(uid)
            if start is not None and end is not None:
                total += abs(end - start) / math.sqrt(gap)
                pairs += 1
            elif start is not None or end is not None:
                total += 0.9 / math.sqrt(gap)   # crossing the edge of the chart is a move
                pairs += 1
        if pairs == 0:
            return floor
        # Shrink toward the chart-wide figure: a few pairs are weak evidence.
        return (total + floor * 3.0) / (pairs + 3.0)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class Forecaster:
    def __init__(self, reads: list[dict], task: dict, params: dict | None = None):
        self.params = dict(PARAMS, **(params or {}))
        self.task = task
        self.reads = reads
        self.places = self._place_count()
        self.hours = window_hours(task, reads)
        self.history = History(reads, self.params) if reads else None

    def _place_count(self) -> int:
        places = self.task.get("places")
        if isinstance(places, int) and not isinstance(places, bool) and 1 <= places <= 10:
            return places
        return 10

    # -- one hourly slot ---------------------------------------------------
    def slot_matrix(self, horizon: float) -> tuple[list[list[float]], dict[int, float]]:
        """Games x (places + 1) probabilities at ``horizon`` hours, place-balanced.

        Returns the matrix and the named entrants with their growth strength.
        Rows are the games carried from the last chart, then the named
        entrants; the last column of each row is "somewhere below the places
        modelled".  Each row sums to 1, and each place's column sums to one
        game minus the probability that place keeps for games not named at all.
        """
        params = self.params
        history = self.history
        assert history is not None

        carried = history.order[: params["max_games"]]
        entrants = self._entrants(carried)
        bounds = log_rank_bounds(self.places)

        rows = [self._chart_row(uid, horizon, bounds) for uid in carried]
        rows.extend(self._entrant_row(horizon, strength) for strength in entrants.values())

        # What each place keeps for games that are nowhere in the history.
        unseen = self._unseen(horizon)
        targets = [1.0 - share for share in unseen]
        targets.append(max(float(len(rows)) - sum(targets), 1e-9))
        self._balance(rows, targets)
        self._mix_scenarios(rows, horizon)
        return rows, entrants

    def _mix_scenarios(self, rows: list[list[float]], horizon: float) -> None:
        """Blend in the two things the chart does that a diffusion alone misses.

        A chart is computed, not a live count, so between recomputes it does
        not move at all -- and when it does recompute, the whole order can
        shift as one game leaves the top.  Both are real, both are cheap to
        mix in, and both make a forecast far less brittle on a quiet day or a
        day when a leader vanishes.
        """
        settings = self.params
        freeze = settings["freeze_weight"] * min(horizon / 24.0, 1.0)
        dropout = settings["dropout_weight"] * min(horizon / 24.0, 1.0)
        if freeze <= 0.0 and dropout <= 0.0:
            return
        places = self.places
        count = len(rows)
        for index, row in enumerate(rows):
            for place in range(places):
                row[place] *= 1.0 - freeze - dropout
            # Frozen: the game stays exactly where it is.
            if freeze > 0.0 and index < places:
                row[index] += freeze
            # A leader left: this game moves up one place.
            if dropout > 0.0 and 0 < index < places:
                row[index - 1] += dropout

    def _chart_row(self, uid: int, horizon: float, bounds: list[float]) -> list[float]:
        """A game on the last chart: drift its log-rank, spread it, lose the edge."""
        params = self.params
        history = self.history
        assert history is not None
        place = history.place[uid]
        centre = history.ranks[uid] + self._drift(uid, horizon)

        narrow = self._width(uid, place, horizon)
        wide = narrow * params["tail_mult"]
        tail = params["tail_weight"]

        row: list[float] = []
        previous = 0.0
        for bound in bounds[1:]:
            share = ((1.0 - tail) * norm_cdf((bound - centre) / narrow)
                     + tail * norm_cdf((bound - centre) / wide))
            share = min(max(share, previous), 1.0)
            row.append(share - previous)
            previous = share
        # What is left is "below the places modelled": the tail the normal has
        # beyond the last bound, plus the chance of slipping off entirely.
        row.append(max(1.0 - sum(row), 1e-9))
        return row

    def _drift(self, uid: int, horizon: float) -> float:
        """Log-rank displacement after ``horizon`` hours (negative is a climb).

        Two measured effects, both from the history:

        * player growth.  The chart ranks the largest *relative increase in
          daily active users*, so a game whose players are growing faster than
          the rest of the chart's is on its way up; the effect builds over
          ``surge_tau`` hours rather than arriving at once.
        * ageing.  The growth that put a game on the chart was measured over
          the past week, so a game that has been climbing for a while has
          already banked most of it and drifts back down.

        Rank momentum -- carrying a game's recent climb or fall forward -- was
        tried and rejected: on this chart mean reversion dominates it, and every
        setting of it scored worse in the backtests.
        """
        params = self.params
        history = self.history
        assert history is not None
        drift = 0.0
        surge = history.surge.get(uid, 0.0)
        if surge >= params["surge_from"]:
            drift -= params["surge_share"] * surge * (
                1.0 - math.exp(-horizon / params["surge_tau"]))
        ageing = (history.tenure.get(uid, 0.0) - history.median_tenure) / 24.0
        if ageing > 0.0:
            drift += params["tenure_drift"] * ageing * min(horizon / 12.0, 1.0)
        return drift

    def _width(self, uid: int, place: int, horizon: float) -> float:
        params = self.params
        history = self.history
        assert history is not None
        width = params["sigma_hour"] * math.sqrt(horizon)
        # The bottom of a top-10 chart churns far more than the top of it.
        width *= 1.0 + params["sigma_place_mult"] * math.log(place) / math.log(self.places + 1)
        width *= history.jitter.get(uid, 1.0)
        ageing = (history.tenure.get(uid, 0.0) - history.median_tenure) / 24.0
        if ageing > 0.0:
            width *= 1.0 + params["tenure_width"] * ageing
        return max(math.sqrt(params["sigma0"] ** 2 + width ** 2), 0.02)

    def _entrant_row(self, horizon: float, strength: float = 1.0) -> list[float]:
        """A game below the top 10: a modest chance of arriving, mostly mid-table."""
        params = self.params
        chance = self._entrant_chance(horizon, strength)
        centre = math.log(params["entrant_center"])
        width = params["entrant_width"]
        weights = [math.exp(-0.5 * ((math.log(place) - centre) / width) ** 2)
                   for place in range(1, self.places + 1)]
        total = sum(weights) or 1.0
        row = [chance * weight / total for weight in weights]
        row.append(max(1.0 - chance, 1e-9))
        return row

    def _entrant_chance(self, horizon: float, strength: float = 1.0) -> float:
        """P(a named game below the chart is in the top 10 within ``horizon`` hours).

        A game whose players are multiplying is likelier to arrive, so the
        chance grows with how far its growth runs past the threshold that got
        it named in the first place.
        """
        params = self.params
        base = 0.30
        chance = base * (1.0 + params["entrant_boost"] * max(strength - 1.0, 0.0))
        return min(0.55, chance) * (1.0 - math.exp(-horizon / 9.0))

    def _entrants(self, carried: list[int]) -> int:
        """Games below the top 10 that are taking off, and so may step in.

        Scanned over the whole chart, not just its top: a breakout can lift a
        game from the far end of the chart into the top 10 within a day, which
        is exactly the event a forecast that looks only at the top misses.
        """
        params = self.params
        history = self.history
        assert history is not None
        depth = params["entrant_depth"]
        threshold = params["entrant_surge"]
        carried_set = set(carried)
        scored = []
        for uid in history.order[self.places: self.places + depth]:
            if uid in carried_set:
                continue       # already has a row of its own
            growth = max(history.surge.get(uid, 0.0), history.surge_fast.get(uid, 0.0))
            if growth < threshold:
                continue
            place = history.place.get(uid, len(history.order))
            # Already close to the top 10 is worth a little extra.
            scored.append((growth - 0.010 * (place - self.places), growth, uid))
        scored.sort(key=lambda item: (-item[0], item[2]))
        # Each named entrant's strength: 1.0 at the threshold, more above it.
        return {uid: max(growth / max(threshold, 1e-6), 1.0)
                for _, growth, uid in scored[: self.params.get("max_entrants", 4)]}

    def _unseen(self, horizon: float) -> list[float]:
        """Per-place probability left for games that are nowhere in the history.

        A game that is not in any read of the history can still be on the chart
        tomorrow -- a brand-new game, or one that was never recorded.  This is
        the only mass such a game gets, and the harness spreads it over such
        games when it scores them, so it is set by backtest, not by intuition.
        """
        params = self.params
        base = min(params["entry_rate"] * horizon, params["entry_cap"])
        return [base * ((place + 1) / 5.5) ** params["entry_slope"]
                for place in range(self.places)]

    # -- balancing ---------------------------------------------------------
    @staticmethod
    def _balance(rows: list[list[float]], targets: list[float]) -> None:
        """Scale rows and columns until each place holds one game's probability.

        Sinkhorn's algorithm for a transportation problem.  Every game must hold
        exactly one place (rows sum to 1) and every place exactly one game
        (columns sum to their target), which is what an ordered chart means.
        """
        if not rows:
            return
        width = len(targets)
        for _ in range(400):
            columns = [0.0] * width
            for row in rows:
                for index, value in enumerate(row):
                    columns[index] += value
            error = 0.0
            for index, total in enumerate(targets):
                if total > 0.0:
                    error = max(error, abs(columns[index] - total) / total)
            if error < 1e-12:
                break
            for index, total in enumerate(targets):
                if columns[index] > 0.0:
                    factor = total / columns[index]
                    for row in rows:
                        row[index] *= factor
            for row in rows:
                total = sum(row)
                if total > 0.0:
                    inverse = 1.0 / total
                    for index in range(width):
                        row[index] *= inverse

    # -- the whole window --------------------------------------------------
    def forecast(self) -> dict:
        if not self.reads or self.history is None:
            raise ValueError("no usable read in the history")

        matrices = [self.slot_matrix(horizon) for horizon in self.hours]
        entrants = matrices[0][1] if matrices else []
        width = self.places + 1
        count = len(matrices[0][0]) if matrices else 0
        average = [[0.0] * width for _ in range(count)]
        for rows, _ in matrices:
            for index, row in enumerate(rows):
                target = average[index]
                for column, value in enumerate(row):
                    target[column] += value / len(matrices)

        games = list(self.history.order)[: self.params["max_games"]] + list(entrants)
        return self._document(games, average)

    def _document(self, games: list[int], matrix: list[list[float]]) -> dict:
        """Write the scored file: named probabilities per place, and the point guess."""
        params = self.params
        places = self.places
        rows: list[dict] = []

        for place in range(1, places + 1):
            column = [(matrix[index][place - 1], uid)
                      for index, uid in enumerate(games)
                      if matrix[index][place - 1] > 0.0]
            column.sort(key=lambda item: (-item[0], item[1]))
            column = column[: params["max_rows_per_place"]]
            total = sum(value for value, _ in column)
            if total <= 0.0:
                continue
            # Keep the meaningful rows, then scale what is left to the share of
            # this place we name.  The rest of a place's probability is the
            # hedge the harness spreads over the games we did *not* name at it:
            # roughly nine of them, so it is a cheap bet on a surprise arrival.
            kept = [(value, uid) for value, uid in column if value / total >= params["min_named_p"]]
            if not kept:
                kept = column[:1]
            mass = sum(value for value, _ in kept)
            share = params["unnamed_share"] / mass
            for value, uid in kept:
                rows.append({"universe_id": uid, "place": place, "p": round(value * share, 7)})
        rows.sort(key=lambda row: (row["place"], -row["p"], row["universe_id"]))
        return {"probabilities": rows, "top10": self._point_guess(games, matrix)}

    def _point_guess(self, games: list[int], matrix: list[list[float]]) -> list[int]:
        """Greedy assignment: each place takes the best game not already used."""
        free = set(range(len(games)))
        guess: list[int] = []
        for place in range(self.places):
            best_index, best_value = None, -1.0
            for index in free:
                value = matrix[index][place]
                if value > best_value or (value == best_value and best_index is not None
                                          and games[index] < games[best_index]):
                    best_value, best_index = value, index
            if best_index is None:
                break
            free.discard(best_index)
            guess.append(games[best_index])
        # A short chart still needs ten distinct ids for a valid file.
        filler = 1
        while len(guess) < self.places:
            while filler in guess:
                filler += 1
            guess.append(filler)
            filler += 1
        return guess


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def persistence(reads: list[dict], places: int) -> dict:
    """The house baseline: the chart does not move.  Only a fallback.

    With no history at all there is nothing to name, so every place is left
    unnamed -- the harness spreads a place over the whole chart, which is the
    only honest forecast when nothing is known.
    """
    places = max(1, min(int(places), 10))
    chart = list(reads[-1]["order"]) if reads else []
    rows = [{"universe_id": uid, "place": index + 1, "p": 0.5}
            for index, uid in enumerate(chart[:places])]
    guess = chart[:places]
    filler = 1
    while len(guess) < places:
        while filler in guess:
            filler += 1
        guess.append(filler)
        filler += 1
    return {"probabilities": rows, "top10": guess}


def main() -> int:
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    task = load_task(os.path.join(in_dir, "task.json"))
    reads = load_reads(os.path.join(in_dir, "history.ndjson"))

    try:
        forecast = Forecaster(reads, task).forecast()
    except Exception as error:                       # noqa: BLE001
        # A broken model must never cost the whole entry: fall back to the
        # house baseline, which at least always scores something.
        sys.stderr.write("model failed (%r); using the persistence baseline\n" % (error,))
        places = task.get("places")
        forecast = persistence(reads, places if isinstance(places, int) else 10)

    os.makedirs(out_dir, exist_ok=True)
    name = task.get("output")
    if not isinstance(name, str) or not name:
        name = "forecast.json"
    path = os.path.join(out_dir, os.path.basename(name))
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(forecast, separators=(",", ":"), ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
