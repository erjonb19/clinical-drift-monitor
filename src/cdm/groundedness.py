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
- **Not numbers:** a run of digits joined to a letter or underscore on either side (``8D``,
  ``site_1``, ``ISIC_0024306``), except the ordinal suffixes ``st``, ``nd``, ``rd`` and ``th``
  (``95th`` is 95). Roman numerals (skin types ``I`` to ``VI``) are not read as numbers.

Anything else fails closed: a number in a form this module does not read (``3.76 × 10^-7``
becomes 3.76, 10 and -7), a date, a derived difference (``0.02 below 0.2``) or a count the
record does not hold is unmatched. Allowed values are every number in the record (numeric
leaves, and numbers written inside its strings) and in the rules context, read the same way.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

# A signed number with optional thousands groups, decimals, exponent, then an optional percent
# sign or ordinal suffix. The lookarounds keep digits glued to letters or underscores out.
NUMBER = re.compile(
    r"(?<![\w.])(?P<sign>[-−])?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dec>\d+))?"
    r"(?:[eE](?P<exp>[-+−]?\d+))?(?:(?P<pct>%)|(?P<ord>st|nd|rd|th)\b)?(?![\w%])"
)
EXACT = 1e-12


@dataclass(frozen=True)
class Number:
    """One number as written, with the tolerance it is allowed."""

    text: str
    value: float
    tolerance: float
    whole: bool


def numbers_in(text: str) -> list[Number]:
    """Every number in ``text``, read under the rules above."""
    found = []
    for m in NUMBER.finditer(text):
        digits = m["int"].replace(",", "")
        dec = m["dec"] or ""
        exp = int(m["exp"].replace("−", "-")) if m["exp"] else 0
        value = float(f"{digits}.{dec}" if dec else digits) * 10.0**exp
        if m["sign"]:
            value = -value
        places = len(dec) - exp
        if m["pct"]:
            value, places = value / 100, places + 2
        whole = not dec and not m["exp"] and not m["pct"]
        tolerance = EXACT if whole else 0.5 * 10.0**-places + EXACT
        found.append(Number(m.group(0), value, tolerance, whole))
    return found


def _leaves(obj: Any) -> Iterator[Any]:
    if isinstance(obj, Mapping):
        for v in obj.values():
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
            values += [n.value for n in numbers_in(leaf)]
    return values


def matches(number: Number, allowed: list[float]) -> bool:
    return any(abs(v - number.value) <= number.tolerance for v in allowed)


def _strings_and_numbers(obj: Any) -> Iterator[str]:
    """Each string leaf, and each numeric leaf written as Python writes it."""
    for leaf in _leaves(obj):
        if isinstance(leaf, str):
            yield leaf
        elif isinstance(leaf, (int, float)) and not isinstance(leaf, bool):
            yield repr(leaf)


def check(
    report: Any, record: Mapping[str, Any], rules: Mapping[str, Any], floor: float
) -> dict[str, Any]:
    """Groundedness of one report (a string, or the report JSON: every string and number in it is
    read). ``grounded`` is true only when every number found matches an allowed value."""
    allowed = allowed_values(record, rules, floor)
    found = [n for s in _strings_and_numbers(report) for n in numbers_in(s)]
    unmatched = sorted({n.text for n in found if not matches(n, allowed)})
    return {"grounded": not unmatched, "numbers": len(found), "unmatched": unmatched}
