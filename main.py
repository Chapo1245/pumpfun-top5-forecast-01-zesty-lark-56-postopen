#!/usr/bin/env python3
"""Top-5 forecaster for new pump.fun coins, 24 hours out.

Entry for make.fun competition ``pumpfun-top5-forecast-01``.  It reads
``$MAKEFUN_INPUT_DIR/history.ndjson`` plus ``task.json`` (defaults
``/input``) and writes ``$MAKEFUN_OUTPUT_DIR/forecast.json`` (default
``/output``) holding exactly ``{probabilities, top5, sizes}``.

The idea
--------
A coin's log10 market cap (in SOL) is carried forward to each of the
seven answer reads from ``task.json`` with a momentum model: the coin's
own recent hourly slope, clipped and shrunk toward a drift prior that
depends on the coin's current level (estimated from the whole history --
here the leaders keep climbing while the mid-pack bleeds), with the
growth saturating over a ~1.5 h time constant and a slow decay once a
coin is more than ~2 h old, so no leader is projected to sit on top
forever.  A fixed-seed Monte Carlo (6,000 draws per read) over the
projected caps, with uncertainty growing with the horizon, gives the
place probabilities and each place's size distribution; the forecast is
averaged over the seven reads because one forecast is scored at every
read.

Newcomers are the hard part: coins born after the history ends took a
third of the practice's top-5 slots and no forecast can name them.  Two
things cover them.  (1) Phantom newcomers compete inside the Monte
Carlo: their caps are drawn around a per-place trend of log-cap against
log-age since the opening, so they steal place mass in proportion to
the newcomer threat, which ramps up with the horizon (0.5 h time
constant) and is fully saturated at the final's 7-13 h horizon.  (2)
Each place's size distribution mixes the Monte Carlo's k-th-largest
distribution with the trend's own wide normal, because the order
statistic of known coins alone is far too narrow and too low at long
horizons.  Named place mass is capped at 0.85, so at least 15% of every
place always stays unnamed for the coins not yet born.

Python 3.13, standard library only, no network, deterministic: the same
input gives byte-identical output.
"""

import json
import math
import os
import random
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- model constants (tuned on the practice holdouts, see README.md) ---
SEED = 20261112
DRAWS_PER_READ = 6000
SATURATION_TAU_H = 1.5       # growth saturates over this many hours
BASE_SIGMA = 0.12            # log10-cap uncertainty at the cut
SIGMA_PER_SQRT_HOUR = 0.22   # extra uncertainty per sqrt(hour) of horizon
MAX_NAMED = 0.85             # cap on a place's named probability mass
MIN_PLACE_P = 0.0008         # smallest named place probability kept
MIN_SIZE_P = 0.0005          # smallest named size-bin probability kept
MAX_ROWS = 100               # RULES.txt cap on rows per place
STALE_DECAY = 0.15           # extra log-cap decay/hr for coins off the chart
SLOPE_CLIP = 1.0             # own recent slope clipped to +-1 log10/hr
OLD_AGE_DECAY = 0.06         # log-cap decay/hr once a coin is past its pump
OLD_AGE_START_H = 2.0        # age (hours) after which the decay starts
OLD_AGE_MAX = 0.6            # cap on the total old-age decay
N_PHANTOMS = 10              # phantom newcomers at a long horizon
PHANTOM_SIGMA = 0.40
PHANTOM_STEP = 0.12          # log10 spacing between phantom means
PHANTOM_TAU_H = 0.5          # horizon ramp for the newcomer threat
TREND_SIZE_W = 0.5           # weight of the trend normal in sizes
TREND_SIZE_TAU_H = 0.5
TREND_SIGMA = 0.40           # sd of the trend normal for a place's size
TREND_SLOPE_CAP = 1.2        # cap on the fitted log-age trend slope


def parse_ts(text):
    """ISO 8601 (with Z) -> epoch seconds."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def logcap_sol(lamports):
    """log10 of a market cap in SOL, as the scorer takes it."""
    return math.log10(int(lamports)) - 9.0


def load_history():
    """Return (task, marks): task.json plus every ok mark with rows."""
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as fh:
        task = json.load(fh)
    marks = []
    with open(os.path.join(INPUT_DIR, "history.ndjson"), encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            mark = json.loads(line)
            if mark.get("ok") and mark.get("rows"):
                marks.append(mark)
    if not marks:
        raise ValueError("the history holds no ok mark with rows")
    return task, marks


def collect_coins(marks):
    """mint -> {"points": [(t, logcap)], "created_at": ts, ...}.

    One point per mark a coin appears in, in time order.
    """
    coins = {}
    for mark in marks:
        t = parse_ts(mark["at"])
        for row in mark["rows"]:
            mint = row["mint"]
            entry = coins.setdefault(
                mint, {"points": [], "created_at": parse_ts(row["created_at"])})
            entry["points"].append((t, logcap_sol(row["market_cap_lamports"])))
    for entry in coins.values():
        entry["points"].sort()
        entry["last_t"] = entry["points"][-1][0]
        entry["last_cap"] = entry["points"][-1][1]
    return coins


def level_drift(coins):
    """Hourly log-cap drift as a function of the coin's level.

    Every consecutive pair of marks a coin appears in contributes one
    drift observation, bucketed by its starting log-cap and shrunk
    toward the global mean when a bucket is thin.  Returns
    (edges, drifts, global_drift).
    """
    edges = [0.0, 1.5, 2.0, 2.5, 3.0, 99.0]
    sums = [0.0] * (len(edges) - 1)
    counts = [0] * (len(edges) - 1)
    gsum = 0.0
    gcount = 0
    for entry in coins.values():
        pts = entry["points"]
        for (t1, c1), (t2, c2) in zip(pts, pts[1:]):
            dt_h = (t2 - t1) / 3600.0
            if dt_h < 0.05:
                continue
            drift = max(-1.5, min(1.5, (c2 - c1) / dt_h))
            gsum += drift
            gcount += 1
            for i in range(len(edges) - 1):
                if edges[i] <= c1 < edges[i + 1]:
                    sums[i] += drift
                    counts[i] += 1
                    break
    global_drift = gsum / gcount if gcount else 0.0
    drifts = []
    for i in range(len(edges) - 1):
        if counts[i]:
            drifts.append((sums[i] + 30.0 * global_drift) / (counts[i] + 30.0))
        else:
            drifts.append(global_drift)
    return edges, drifts, global_drift


def drift_for(level, edges, drifts):
    for i in range(len(edges) - 1):
        if edges[i] <= level < edges[i + 1]:
            return drifts[i]
    return drifts[-1]


def project(coins, edges, drifts, read_ts, last_mark_ts, ceiling_cap):
    """Project every coin's log-cap to one answer read.

    Returns a list of (mint, mean_logcap, sigma), keeping only coins
    that could plausibly reach the top: within 1.2 log of the best
    projection, at most 160 of them.
    """
    ceiling = ceiling_cap + 1.0
    out = []
    for mint, entry in coins.items():
        pts = entry["points"]
        last_t, last_c = pts[-1]
        dt_h = max(0.0, (read_ts - last_t) / 3600.0)
        # Own recent slope from the last two marks, clipped; shrunk
        # toward the level prior.  A single observation -> prior alone.
        own = None
        if len(pts) >= 2:
            t_prev, c_prev = pts[-2]
            span_h = (last_t - t_prev) / 3600.0
            if span_h >= 0.08:
                own = max(-SLOPE_CLIP,
                          min(SLOPE_CLIP, (last_c - c_prev) / span_h))
        prior = drift_for(last_c, edges, drifts)
        slope = (0.65 * own + 0.35 * prior) if own is not None else prior
        # Saturating growth: slope * TAU * (1 - exp(-dt/TAU)).
        mean = last_c + slope * SATURATION_TAU_H * (
            1.0 - math.exp(-dt_h / SATURATION_TAU_H))
        # Old coins fade: once a coin is past OLD_AGE_START_H hours old
        # its cap decays, so a leader from the history cannot be
        # projected to hold the top forever.  This bites only at the
        # final's 7-13 h horizon; practice reads are too soon.
        age_h = (read_ts - entry["created_at"]) / 3600.0
        mean -= min(OLD_AGE_MAX,
                    OLD_AGE_DECAY * max(0.0, age_h - OLD_AGE_START_H))
        # Coins that fell off the chart are fading: extra decay.
        stale_h = max(0.0, (last_mark_ts - last_t) / 3600.0)
        if stale_h > 0.5:
            mean -= min(0.6, STALE_DECAY * (stale_h - 0.5))
        mean = min(ceiling, max(0.5, mean))
        sigma = math.sqrt(BASE_SIGMA ** 2
                          + (SIGMA_PER_SQRT_HOUR ** 2) * dt_h
                          + (0.10 * stale_h) ** 2)
        out.append((mint, mean, sigma))
    out.sort(key=lambda item: (-item[1], item[0]))
    if out:
        best = out[0][1]
        out = [item for item in out if item[1] >= best - 1.2][:160]
    return out


def place_trends(marks, t0_ts, places):
    """Per-place log-cap trend against log-age since the opening.

    For each place k, regress the log10 cap of the coin at #k in each
    mark on log10(age in minutes).  New winners arrive steadily, so
    every place's cap drifts up with age; the log-age form makes that
    drift self-limiting as the market matures.  Returns a list of
    (intercept, slope) per place; a place with too little data falls
    back to a flat trend at its last observed cap.
    """
    series = {k: [] for k in range(1, places + 1)}
    for mark in marks:
        age_min = max(1.0, (parse_ts(mark["at"]) - t0_ts) / 60.0)
        chart = sorted(mark["rows"], key=lambda r: r["rank"])
        for k in range(1, places + 1):
            if k - 1 < len(chart):
                series[k].append(
                    (math.log10(age_min),
                     logcap_sol(chart[k - 1]["market_cap_lamports"])))
    trends = []
    for k in range(1, places + 1):
        pts = series[k]
        if len(pts) < 2:
            level = pts[0][1] if pts else 2.0
            trends.append((level, 0.0))
            continue
        n = len(pts)
        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        slope = 0.0
        if sxx > 1e-9:
            slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
        slope = max(0.0, min(TREND_SLOPE_CAP, slope))
        trends.append((my - slope * mx, slope))
    return trends


def bin_of(value, edges):
    """Size bin 0..len(edges) for a log10 cap in SOL (RULES.txt)."""
    if value < edges[0]:
        return 0
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges)


def normal_bin_p(mean, sd, edges, b):
    """P(a Normal(mean, sd) falls in size bin b)."""
    lo = -math.inf if b == 0 else edges[b - 1]
    hi = math.inf if b == len(edges) else edges[b]

    def cdf(x):
        if x == -math.inf:
            return 0.0
        if x == math.inf:
            return 1.0
        return 0.5 * (1.0 + math.erf((x - mean) / (sd * math.sqrt(2.0))))

    return cdf(hi) - cdf(lo)


def floor6(p):
    return math.floor(p * 1_000_000) / 1_000_000


def main():
    task, marks = load_history()
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    reads = [parse_ts(r) for r in task["reads"]]
    last_mark_ts = max(parse_ts(m["at"]) for m in marks)

    coins = collect_coins(marks)
    dedges, drifts, _global = level_drift(coins)
    max_cap = max(e["last_cap"] for e in coins.values())
    t0_ts = parse_ts(task["t0"])
    trends = place_trends(marks, t0_ts, places)

    rng = random.Random(SEED)
    # place_counts[place][mint] counts over all reads; size_hists holds
    # one k-th-largest histogram per read, plus the trend's prediction
    # and weight per read for the size mixture.
    place_counts = {k: {} for k in range(1, places + 1)}
    size_hists = []
    trend_weights = []
    trend_at_read = []
    mean_cap = {}  # mint -> mean projected log-cap across reads (tie-break)

    for read_ts in reads:
        projected = project(coins, dedges, drifts, read_ts, last_mark_ts,
                            max_cap)
        for mint, mean, _sigma in projected:
            mean_cap[mint] = mean_cap.get(mint, 0.0) + mean / len(reads)
        dt_h = max(0.0, (read_ts - last_mark_ts) / 3600.0)
        age_min = max(1.0, (read_ts - t0_ts) / 60.0)
        trend = [a + b * math.log10(age_min) for (a, b) in trends]
        trend_at_read.append(trend)
        # The newcomer threat grows with the horizon: a coin born just
        # after the cut has had no time to grow at a 10-minute horizon,
        # but over 7-13 hours newcomers routinely take top places.
        ramp = 1.0 - math.exp(-dt_h / PHANTOM_TAU_H)
        n_phantom = max(1, round(N_PHANTOMS * ramp))
        trend_weights.append(TREND_SIZE_W *
                             (1.0 - math.exp(-dt_h / TREND_SIZE_TAU_H)))
        # Phantom newcomers: coins born after the history ends, centred
        # on the trend's place-1 prediction and spread below it.  They
        # compete in the Monte Carlo but are never named in the output.
        entries = [(mean, sigma, mint) for mint, mean, sigma in projected]
        for j in range(n_phantom):
            entries.append((trend[0] - j * PHANTOM_STEP, PHANTOM_SIGMA, None))
        size_hist = {k: {} for k in range(1, places + 1)}
        for _ in range(DRAWS_PER_READ):
            drawn = [(rng.gauss(mean, sigma), mint)
                     for mean, sigma, mint in entries]
            drawn.sort(key=lambda item: (-item[0], item[1] or ""))
            for k in range(1, places + 1):
                if k - 1 < len(drawn):
                    value, mint = drawn[k - 1]
                    if mint is not None:
                        place_counts[k][mint] = \
                            place_counts[k].get(mint, 0) + 1
                    b = bin_of(value, edges)
                    size_hist[k][b] = size_hist[k].get(b, 0) + 1
        size_hists.append(size_hist)

    total_draws = DRAWS_PER_READ * len(reads)
    probabilities = []
    for k in range(1, places + 1):
        ranked = sorted(place_counts[k].items(),
                        key=lambda kv: (-kv[1], kv[0]))
        rows = []
        for mint, count in ranked[:MAX_ROWS]:
            p = floor6(count / total_draws)
            if p >= MIN_PLACE_P:
                rows.append({"mint": mint, "place": k, "p": p})
        # Cap the named mass so part of every place stays unnamed.
        total = sum(r["p"] for r in rows)
        if total > MAX_NAMED:
            scale = MAX_NAMED / total
            rows = [{"mint": r["mint"], "place": k,
                     "p": floor6(r["p"] * scale)} for r in rows]
            rows = [r for r in rows if r["p"] >= MIN_PLACE_P]
        probabilities.extend(rows)

    sizes = []
    for k in range(1, places + 1):
        # Mixture, averaged over reads: the Monte Carlo's k-th-largest
        # distribution (which now includes phantom newcomers) and the
        # trend's wide normal, weighted by horizon.
        for b in range(len(edges) + 1):
            ps = []
            for r in range(len(reads)):
                w = trend_weights[r]
                mc_p = size_hists[r][k].get(b, 0) / DRAWS_PER_READ
                trend_p = normal_bin_p(trend_at_read[r][k - 1], TREND_SIGMA,
                                       edges, b)
                ps.append((1.0 - w) * mc_p + w * trend_p)
            p = floor6(sum(ps) / len(ps))
            if p >= MIN_SIZE_P:
                sizes.append({"place": k, "bin": b, "p": p})

    # top5: highest expected agreement weight, tie-break by mean projection.
    weight = {}
    for row in probabilities:
        w = (places + 1 - row["place"]) * row["p"]
        weight[row["mint"]] = weight.get(row["mint"], 0.0) + w
    top5 = sorted(weight,
                  key=lambda m: (-weight[m], -mean_cap.get(m, 0.0), m))
    top5 = top5[:places]
    # Safety: exactly `places` distinct mints, fall back to best projections.
    if len(top5) < places:
        extra = sorted(mean_cap, key=lambda m: (-mean_cap[m], m))
        for mint in extra:
            if mint not in top5:
                top5.append(mint)
            if len(top5) == places:
                break

    probabilities.sort(key=lambda r: (r["place"], -r["p"], r["mint"]))
    sizes.sort(key=lambda r: (r["place"], r["bin"]))
    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w",
              encoding="utf-8") as fh:
        fh.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
