import json
import re
from pathlib import Path

import pandas as pd

from data_extraction.normalize import parse_numeric


AMBIGUOUS = "ambiguous"
# "model"/"mpn" guard part/model-number fields: unit-aware parsing can pull a
# number out of "WH-1000XM4", which must not turn a key into a numeric measure
ID_TOKENS = {"id", "ean", "asin", "upc", "serial", "barcode", "isbn", "gtin", "sku",
             "code", "model", "mpn"}
COUNT_RE = re.compile(r"number of|quantity|\bcount\b|\bqty\b|nombre|numero", re.I)


def _normalize(name):
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def detect_type(series, name="", uniq_ratio=0.0):
    s = series.dropna()
    if s.empty:
        return "categorical", 1.0

    # url/link columns (image_url, ...) are keys, not measurements: to_number
    # would pull a stray digit out of the URL and scale it. Filter by name.
    if "url" in _normalize(name).split():
        return "identifier", 0.99

    n_unique = int(s.nunique())
    # unit-aware parsing (normalize.to_number): "40mm", "20 hrs", "$1,299.00",
    # "5 hz - 35.000 hz" all yield a number, so measure-like columns that
    # pd.to_numeric cannot read are still recognized as numeric
    parsed = parse_numeric(s)
    numeric_frac = float(parsed.notna().mean())
    vals = parsed.dropna()
    is_float_measure = numeric_frac >= 0.98 and float((vals % 1 != 0).mean()) > 0.02
    norm = _normalize(name)

    # identifiers: pure keys or high-cardinality coded fields, never continuous measures
    if not is_float_measure:
        if uniq_ratio >= 0.98:                       # near-unique key, no clustering signal
            return "identifier", 0.95
        if set(norm.split()) & ID_TOKENS and n_unique > 100:
            return "identifier", 0.9

    if numeric_frac < 0.98:
        return "categorical", 0.99
    if is_float_measure:
        return "numeric", 0.99

    # all-integer numeric: count semantics first, then cardinality
    if COUNT_RE.search(norm):
        return "numeric", 0.9
    if n_unique <= 10:
        return "categorical", 0.9
    if n_unique >= 50:
        return "numeric", 0.85
    return AMBIGUOUS, 0.0


def _build_prompt(payloads):
    schema = ('{"features": [{"feature": str, "type": "categorical" | "numeric", '
              '"confidence": number 0-1, "reason": str}]}')
    lines = [
        "Classify each product feature as categorical or numeric for a clustering pipeline.",
        "categorical = discrete labels, codes, or ordinal buckets.",
        "numeric = continuous measurements or open-ended counts.",
        "Decide from the feature name and the sample values. Respond ONLY as JSON matching:",
        schema,
        "",
        "Features:",
        json.dumps(payloads, indent=2),
    ]
    return "\n".join(lines)


class FeatureType:
    def __init__(self, cache_path,
                 ollama_model="qwen2.5:14b", ollama_url="http://localhost:11434",
                 confidence_floor=0.6):
        self.cache_path = Path(cache_path)
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url.rstrip("/")
        self.confidence_floor = confidence_floor

    def resolve(self, df):
        cache = self._load_cache()

        pending = {}
        verdicts = {}
        for feat in pd.unique(df.columns.get_level_values(1)):
            key = str(feat)
            if key in cache:
                verdicts[key] = cache[key]
                continue

            sub = df.loc[:, df.columns.get_level_values(1) == feat]
            pooled = pd.Series(sub.to_numpy().ravel())
            uniq_ratio = float((sub.nunique() / len(df)).max())
            kind, conf = detect_type(pooled, name=key, uniq_ratio=uniq_ratio)
            if kind == AMBIGUOUS:
                pending[key] = self._payload(key, df, pooled)
            else:
                verdicts[key] = {"type": kind, "confidence": conf, "source": "rule"}

        if pending:
            verdicts.update(self._resolve_ambiguous(pending))

        cache.update(verdicts)
        self._save_cache(cache)
        return {k: v["type"] for k, v in verdicts.items()}

    def _resolve_ambiguous(self, pending):
        llm = {}
        try:
            llm = self._classify_ollama(list(pending.values()))
        except Exception as e:
            print(f"[typer] local LLM unavailable ({e})")

        out = {}
        for key in pending:
            v = llm.get(key) or {"type": "categorical", "confidence": 0.0, "source": "fallback"}
            if v["confidence"] < self.confidence_floor:
                print(f"[typer] low confidence on '{key}' ({v['confidence']}); review in cache")
            out[key] = v
        return out

    def _payload(self, feat, df, pooled):
        mask = df.columns.get_level_values(1) == feat
        provenance = [str(p) for p in df.columns[mask].get_level_values(0).unique()]
        parsed = parse_numeric(pooled)
        return {
            "feature": feat,
            "provenance": provenance,
            "samples": [str(v) for v in pooled.dropna().unique()[:15]],
            "n_unique": int(pooled.nunique()),
            "numeric_frac": round(float(parsed.notna().mean()), 3),
        }

    def _classify_ollama(self, payloads, batch_size=8):
        # small batches: one oversized prompt makes the model drift off the
        # response schema (observed as a missing "features" root), and a single
        # bad reply would otherwise sink every pending feature at once
        import requests

        out = {}
        for i in range(0, len(payloads), batch_size):
            batch = payloads[i:i + batch_size]
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.ollama_model, "prompt": _build_prompt(batch),
                      "format": "json", "stream": False, "options": {"temperature": 0}},
                timeout=180,
            )
            resp.raise_for_status()
            data = json.loads(resp.json()["response"])
            items = data.get("features", data if isinstance(data, list) else [])
            for item in items:
                out[item["feature"]] = {
                    "type": item["type"],
                    "confidence": float(item.get("confidence", 0.5)),
                    "reason": item.get("reason", ""),
                    "source": "ollama",
                }
        return out

    def _load_cache(self):
        if self.cache_path.exists():
            return json.loads(self.cache_path.read_text())
        return {}

    def _save_cache(self, cache):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
