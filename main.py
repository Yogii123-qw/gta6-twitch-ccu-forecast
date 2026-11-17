"""Forecast GTA VI's launch-day Twitch viewers hour by hour (quantiles).

Method (measured, not guessed): kernel-weighted lognormal ratios over the
case's own comparables -- past big game launches' release-day curves.

For each hour h = 2..24, every donor launch j with a complete 24-hour
release-day curve contributes its log-ratio ln(v_h / v_1). Donors are weighted
by how close their hour-1 print is to this case's hour-1 print p1, with a
Gaussian kernel in log space (bandwidth 1.2), and donors whose hour 1 was a
pre-unlock trickle (h1 < 5% of their own day peak: simultaneous-global
unlocks, whose jump no hour-1 method can foresee) are left out. Per hour,
the weighted mean mu_h and weighted standard deviation sigma_h of the
log-ratios give the quantiles q_tau = p1 * exp(mu_h + z_tau * sigma_h).
The median curve and the log-offsets of the other quantiles are smoothed
with a 3-hour moving average, so one noisy donor hour cannot kink a row.

Why this shape: a midnight-local console launch (GTA VI's unlock) has its
hour 1 already substantial -- UK just unlocked, NZ/AU/JP already playing --
then dips through the small hours and climbs as the US unlocks (05:00/08:00
UTC) into the US/Europe evening. A flat persistence median misses that
ramp by ~80%; the donors' median ratio rises from ~0.75 at hours 3-10 to
~1.8 at hour 23. The lognormal spread is honest about the remaining
uncertainty: on held-out launches the 5%-95% band covers ~0.89 of hours.

Measured by leave-one-out cross-validation over the kit's comparables
(eval.py/eval2.py/eval3.py/eval4.py in the build notes; scorer anchored on
the RULES worked example, loss 0.112545): mean pinball loss on the 29
midnight-local-like launches, lower is better --
  house persistence baseline .... 0.3595 (median 0.2619)
  unweighted empirical quantiles . 0.3888 (global-unlock donors poison it)
  kernel bw=1.5, donors >= 5% ... 0.2683
  THIS program (lognormal, bw=1.2)  0.2638 (median 0.1883, typical miss 57%,
                                     90% coverage 0.885)
The house baseline's own practice loss is ~0.387. Generations past this
point moved the LOO mean by <=0.002 (noise), so iteration stopped here.

Fallbacks keep every case valid: with no usable donor the program falls
back to the house persistence formula; a missing hour-1 print falls back
to the last pre-checkpoint history point (floored at 1 viewer). Output is
deterministic: no randomness, no time, no network, stdlib only.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (none is needed today).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Gaussian kernel bandwidth (log units) on donor hour-1 prints.
BANDWIDTH = 1.2
# Donors whose hour 1 is below this share of their own day peak are
# pre-unlock trickles of a simultaneous-global launch: excluded.
MIN_H1_PEAK = 0.05
# Moving-average window (hours) for smoothing the median curve and spreads.
SMOOTH_WINDOW = 3
# Persistence fallback spread (same 0.15 as the house baseline).
FALLBACK_SIGMA = 0.15


def round2(x):
    """x rounded to 2 decimals, half up (same as the starter's round2)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def release_day_curve(comp):
    """{hour: viewers} for hours 1..24 of the comparable's UTC release day.

    Returns None unless all 24 hours are present with positive viewers.
    The GTA VI category's own pre-release history fails this (its day is
    the Aug-27 premiere, with only a few hours on it), so it never poses
    as a launch curve.
    """
    try:
        day_start = datetime.fromisoformat(
            comp["day_start"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None
    curve = {}
    for row in comp.get("hourly") or []:
        try:
            when = datetime.fromisoformat(row["from"].replace("Z", "+00:00"))
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None or viewers <= 0:
            continue
        hour = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            curve[hour] = float(viewers)
    if len(curve) != 24 or curve.get(1, 0) <= 0:
        return None
    return curve


def fallback_forecast(p1, hours=range(2, 25)):
    """House persistence baseline: every hour stays at hour 1's print."""
    p1 = max(1.0, float(p1))
    out = []
    for h in hours:
        row = [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1)))
               for z in Z]
        out.append({"hour": h, "q": row})
    return {"hours": out}


def smooth(values, hours):
    """Moving average of a {hour: log value} curve over the given hours.

    Edge-aware: hour 2 averages hours 2-3, the last hour its last two.
    Neighbours are taken from `hours`, so a case scoring a subset of
    hours still smooths over the hours it actually has.
    """
    out = {}
    half = SMOOTH_WINDOW // 2
    hs = sorted(hours)
    for h in hs:
        lo = max(hs[0], h - half)
        hi = min(hs[-1], h + half)
        seg = [values[k] for k in range(lo, hi + 1) if k in values]
        out[h] = sum(seg) / len(seg)
    return out


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("case asks for levels %r, not %r"
                         % (case.get("levels"), LEVELS))
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is not None and first.get("viewers"):
        p1 = max(1.0, float(first["viewers"]))
    else:
        # No hour-1 print: fall back to the last pre-checkpoint history
        # point, so the case still gets a valid forecast.
        history = case.get("history") or []
        p1 = max(1.0, float(history[-1]["viewers"])) if history else 1.0

    # Hours to forecast come from the case file itself (normally 2..24);
    # falling back to 2..24 keeps a malformed forecast list valid.
    wanted = sorted({row["hour"] for row in case.get("forecast") or []})
    wanted = [h for h in wanted if 2 <= h <= 24] or list(range(2, 25))

    donors = []
    for comp in case.get("comparables") or []:
        curve = release_day_curve(comp)
        if curve is None:
            continue
        if curve[1] / max(curve.values()) < MIN_H1_PEAK:
            continue
        donors.append(curve)
    if not donors:
        return fallback_forecast(p1, wanted)

    log_p1 = math.log(p1)
    weights = [math.exp(-((math.log(curve[1]) - log_p1) / BANDWIDTH) ** 2)
               for curve in donors]

    hours = []
    medians = {}
    offsets = {i: {} for i in range(5)}
    for h in wanted:
        pairs = [(math.log(curve[h] / curve[1]), w)
                 for curve, w in zip(donors, weights)]
        total = sum(w for _, w in pairs)
        mu = sum(v * w for v, w in pairs) / total
        var = sum(w * (v - mu) ** 2 for v, w in pairs) / total
        sigma = math.sqrt(max(var, 1e-9))
        medians[h] = mu
        for i, z in enumerate(Z):
            offsets[i][h] = z * sigma
    medians = smooth(medians, wanted)
    offsets = {i: smooth(offsets[i], wanted) for i in range(5)}
    for h in wanted:
        row = [round2(max(1.0, p1 * math.exp(medians[h] + offsets[i][h])))
               for i in range(5)]
        for i in range(1, 5):
            row[i] = max(row[i], row[i - 1])
        hours.append({"hour": h, "q": row})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
