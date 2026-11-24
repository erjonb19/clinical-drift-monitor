"""Phase 4 (backlog item 8): the groundedness check for drift incident reports.

``results/phase4/eval_spec.json`` (thresholds approved by the owner, 2026-11-24): every number
in a report's prose and claims must match a value in its scenario record or in the fixed rules
context (``scenarios.json``'s ``rules`` and ``accuracy_floor``). A report with any unmatched
number is ungrounded, and groundedness is a hard gate. The spec left the rounding and percent
forms to be fixed here, before any live call. They are:

- **Written with decimals** (``0.184``, ``0.6213``): matches an allowed value ``v`` when
  ``|v - x| <= 0.5 * 10**-d`` for ``d`` written decimals, so either rounding convention passes
  and truncating one digit too far does not.
- **Written as a whole number** (``300``, ``8``): must equal an allowed value exactly. A whole
  number is never a rounding of a fraction, so ``1`` does not match ``0.6``.
- **Percent** (``14%``, ``15.7%``): read as ``x / 100`` with two more decimals, so ``14%``
  matches 0.14 within 0.005 and ``15.7%`` matches 0.157 within 0.0005.
- **Scientific** (``3.76e-07``): matches within half a unit of the mantissa's last written digit.
- **Thousands separators** (``553,019``) are read as one number.
- **Not numbers:** an identifier of letters and digits with no decimal point (``site_1``,
  ``ISIC_0024306``, ``v2``) and the literal ``8D``. Ordinals (``95th``) are read as numbers.
  Roman numerals (skin types ``I`` to ``VI``) contain no digit and are not read.

Every other run of characters that holds a digit must read in full as one of the forms above,
or it is **unmatched**: the check fails closed. So ``0.9x``, ``301images``, ``PSI0.9``, a date
(``2026-11-24``), a range (``7-8``), ``3.76 x 10^-7`` (``-7`` is unmatched), a number too large to
read, a numeral that is not a plain digit (``½``, ``²``), and a ``NaN`` or infinite value in the
report JSON are all unmatched, as is a derived
difference (``0.02 below 0.2``) or a count the record does not hold. ``14 %`` (with a space) is
read as the whole number 14. Dictionary keys are read as well as values.

**What it does not check:** a number is matched against every allowed value, not against the
field it describes, so ``PSI 0.2`` passes because 0.2 is a rule threshold, and a coarse rounding
(``0.2`` for 0.184) passes. Whether each number is the right one is the correctness scoring's
job (the structured findings and the claims' named fields), not this check's.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

# A maximal run of characters holding a digit; each must then read in full as one number.
TOKEN = re.compile(r"[\w.,%+\-−]*\d[\w.,%+\-−]*")
NUMBER = re.compile(
    r"(?P<sign>[-−+])?(?P<int>\d{1,3}(?:,\d{3})+|\d+)?(?:\.(?P<dec>\d+))?"
    r"(?:[eE](?P<exp>[-+−]?\d+))?(?:(?P<pct>%)|(?P<ord>st|nd|rd|th))?"
)
IDENTIFIER = re.compile(r"[A-Za-z]+_?\d+|8D")
EXACT = 1e-12


@dataclass(frozen=True)
class Number:
    """One number as written, with the tolerance it is allowed (``value`` is NaN when the
    token could not be read, which never matches)."""

    text: str
    value: float
    tolerance: float
    whole: bool


def _read(token: str) -> Number | None:
    """The token as a number, ``None`` if it is an identifier, or an unreadable Number."""
    core = token.rstrip(".,")
    if core.startswith(".."):
        core = core.lstrip(".")
    core = core.lstrip(",")
    if IDENTIFIER.fullmatch(core):
        return None
    m = NUMBER.fullmatch(core)
    if not m or not (m["int"] or m["dec"]) or (m["ord"] and (m["dec"] or m["exp"])):
        return Number(core, math.nan, 0.0, False)
    digits = (m["int"] or "0").replace(",", "")
    dec = m["dec"] or ""
    try:
        exp = int(m["exp"].replace("−", "-")) if m["exp"] else 0
        value = float(f"{digits}.{dec}" if dec else digits) * 10.0**exp
    except (OverflowError, ValueError):
        return Number(core, math.nan, 0.0, False)
    if not math.isfinite(value):
        return Number(core, math.nan, 0.0, False)
    if m["sign"] in ("-", "−"):
        value = -value
    places = len(dec) - exp
    if m["pct"]:
        value, places = value / 100, places + 2
    whole = not dec and not m["exp"] and not m["pct"]
    tolerance = EXACT if whole else 0.5 * 10.0**-places + EXACT
    return Number(core, value, tolerance, whole)


def numbers_in(text: str) -> list[Number]:
    """Every number in ``text``, read under the rules above; unreadable tokens included."""
    found = [n for t in TOKEN.findall(text) if (n := _read(t)) is not None]
    # Numerals that are not ASCII-style digits (½, ², ⁷, Roman numeral signs) are not read.
    found += [Number(c, math.nan, 0.0, False) for c in TOKEN.sub(" ", text) if c.isnumeric()]
    return found


def _leaves(obj: Any) -> Iterator[Any]:
    """Every leaf, dictionary keys included."""
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            yield k
            yield from _leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _leaves(v)
    else:
        yield obj


def allowed_values(
    record: Mapping[str, Any], rules: Mapping[str, Any], floor: float
) -> list[float]:
    """Every number the report may use: the record's and the rules context's."""
    values = [float(floor)]
    for leaf in _leaves([record, rules]):
        if isinstance(leaf, bool) or leaf is None:
            continue
        if isinstance(leaf, (int, float)):
            if math.isfinite(leaf):
                values.append(float(leaf))
        elif isinstance(leaf, str):
            values += [n.value for n in numbers_in(leaf) if math.isfinite(n.value)]
    return values


def matches(number: Number, allowed: list[float]) -> bool:
    return any(abs(v - number.value) <= number.tolerance for v in allowed)


def _report_numbers(obj: Any) -> Iterator[Number]:
    """Each number in the report: in its strings, and each numeric leaf as Python writes it (a
    NaN or infinite leaf is unmatched)."""
    for leaf in _leaves(obj):
        if isinstance(leaf, str):
            yield from numbers_in(leaf)
        elif isinstance(leaf, (int, float)) and not isinstance(leaf, bool):
            if not math.isfinite(leaf):
                yield Number(repr(leaf), math.nan, 0.0, False)
            else:
                yield from numbers_in(repr(leaf))


def check(
    report: Any, record: Mapping[str, Any], rules: Mapping[str, Any], floor: float
) -> dict[str, Any]:
    """Groundedness of one report (a string, or the report JSON: every string and number in it is
    read). ``grounded`` is true only when every number found matches an allowed value."""
    allowed = allowed_values(record, rules, floor)
    found = list(_report_numbers(report))
    unmatched = sorted({n.text for n in found if not matches(n, allowed)})
    return {"grounded": not unmatched, "numbers": len(found), "unmatched": unmatched}
