"""Financial amounts and unit declarations."""

from __future__ import annotations

import math
import re
from typing import Any

UNIT_MULTIPLIERS = {
    f"{unit}{suffix}": multiplier
    for unit, multiplier in [
        ("thousand", 1_000.0),
        ("million", 1_000_000.0),
        ("billion", 1_000_000_000.0),
        ("trillion", 1_000_000_000_000.0),
    ]
    for suffix in ("", "s")
}

_SCALES = r"thousands?|millions?|billions?|trillions?"
_CURRENCIES = r"USD|EUR|GBP|JPY|CAD|AUD|CHF|CNY|HKD|US\$|CA\$|C\$|A\$|HK\$|[$€£¥]"
_CURRENCY_RE = re.compile(rf"(?<![a-z])(?:{_CURRENCIES})(?![a-z])", re.I)
SCALE_RE = re.compile(rf"\b({_SCALES})\b", re.I)
_TABLE_SCALE_RE = re.compile(
    rf"(?:^|[(,;\s])(?:\(\s*)?(?:({_CURRENCIES})\s+)?in\s+(?:({_CURRENCIES})\s*)?({_SCALES})\b",
    re.I,
)
_TABLE_CAPTION_SCALE_RE = re.compile(
    rf"(?:^|\()\s*(?:{_CURRENCIES}|dollars?)\s+({_SCALES})\b"
    rf"|(?:^|\()\s*({_SCALES})\s+of\s+(?:dollars?|shares?|{_CURRENCIES})\b",
    re.I,
)
_CURRENCY_SYMBOLS = {
    "US$": "USD",
    "CA$": "CAD",
    "C$": "CAD",
    "A$": "AUD",
    "HK$": "HKD",
    "€": "EUR",
    "£": "GBP",
}
_NUMBER = r"[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
_FOOTNOTE = re.compile(r"(?<=[\d%)])\s*(?:\[\w+\]|\(\s*[\w*]+\s*\)|[*†‡#])+\s*$")
_SIGNED_CURRENCY = re.compile(rf"^([+-])\s*((?:{_CURRENCIES})\s*)", re.I)
_NUMERIC = re.compile(
    rf"(?P<number>{_NUMBER})\s*(?P<percent>%)?\s*(?P<scale>{_SCALES})?",
    re.I,
)
_REPEATED_PERCENT = re.compile(rf"({_NUMBER})\s*%\s*\(\s*{_NUMBER}\s*%\s*\)")


def currency_hints(text: str) -> set[str]:
    matches = {m.upper() for m in _CURRENCY_RE.findall(text)}
    return {_CURRENCY_SYMBOLS.get(m, m) for m in matches}


def scale_hints(text: str, *, declaration_only: bool = False) -> set[str]:
    if declaration_only:
        matches = [match.group(3) for match in _TABLE_SCALE_RE.finditer(text)]
        matches.extend(
            match.group(1) or match.group(2)
            for match in _TABLE_CAPTION_SCALE_RE.finditer(text)
        )
    else:
        matches = SCALE_RE.findall(text)
    return {match.lower().rstrip("s") for match in matches}


def parse_amount(value: Any) -> tuple[float | None, bool, str | None]:
    if value is None or isinstance(value, bool):
        return None, False, None
    if isinstance(value, int | float):
        try:
            number = float(value)
        except OverflowError:
            return None, False, None
        return (number, False, None) if math.isfinite(number) else (None, False, None)

    text = str(value).replace("\u2212", "-").strip()
    text = _FOOTNOTE.sub("", text)
    text = _SIGNED_CURRENCY.sub(r"\2\1", text)
    prefix = re.match(rf"(?:{_CURRENCIES})\s*", text, re.I)
    if prefix:
        text = text[prefix.end() :]

    duplicate = _REPEATED_PERCENT.fullmatch(text)
    if duplicate:
        number = float(duplicate.group(1).replace(",", ""))
        return (number, True, None) if math.isfinite(number) else (None, False, None)

    accounting = text.startswith("(")
    external_percent = accounting and text.endswith(")%")
    if accounting and (external_percent or text.endswith(")")):
        text = text[1 : -2 if external_percent else -1].strip()

    text = re.sub(rf"^(?:{_CURRENCIES})\s*|\s+(?:{_CURRENCIES})$", "", text, flags=re.I)
    text = re.sub(r"^([-+])\s+", r"\1", text)
    match = _NUMERIC.fullmatch(text)
    if not match:
        return None, False, None
    number = float(match.group("number").replace(",", ""))
    if not math.isfinite(number):
        return None, False, None
    if accounting:
        number = -abs(number)
    scale = match.group("scale")
    return (
        number,
        bool(match.group("percent") or external_percent),
        (scale.lower().rstrip("s") if scale else None),
    )


def detect_unit_label(*texts: Any) -> str:
    for text in texts:
        scales = scale_hints(str(text or ""))
        if scales:
            return next(iter(scales)) if len(scales) == 1 else "ambiguous"
    return "base"


def normalize_value(value: float | None, unit_label: str) -> float | None:
    if value is None or unit_label == "ambiguous":
        return None
    normalized = value * UNIT_MULTIPLIERS.get(unit_label.lower(), 1.0)
    return normalized if math.isfinite(normalized) else None


def normalize_with_context(
    value: float | None, *context_texts: Any
) -> dict[str, str | float | None]:
    unit = detect_unit_label(*context_texts)
    return {"unit_label": unit, "value_base": normalize_value(value, unit)}
