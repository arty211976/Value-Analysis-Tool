#!/usr/bin/env python3
"""
Perplexity Client (API)

Fetches up to N years of EPS/ROE fundamentals via Perplexity API and returns a
standardized JSON structure suitable for reconciliation with annual report data.

Notes:
- ASCII-only strings and comments.
- Perplexity output is constrained to strict JSON by response_format schema.
- EPS units preserved; if cents, convert to dollars once.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import requests


logger = logging.getLogger(__name__)


class PerplexityClient:
    """Thin wrapper around Perplexity chat completions for fundamentals."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, timeout_seconds: int = 45):
        # Load .env locally if present so subprocesses pick up API key/settings
        try:
            env_path = os.path.join(os.getcwd(), ".env")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.lstrip("\ufeff")  # strip UTF-8 BOM if present
                        s = line.strip()
                        if not s or s.startswith("#") or "=" not in s:
                            continue
                        if s.lower().startswith("export "):
                            s = s[7:].strip()
                        k, v = s.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        # strip inline comments
                        hash_idx = v.find("#")
                        if hash_idx != -1:
                            v = v[:hash_idx].strip()
                        v = v.strip('"').strip("'")
                        if k and v and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.model_name = model_name or os.getenv("PERPLEXITY_MODEL", "sonar-pro")
        self.timeout_seconds = timeout_seconds
        self.api_url = os.getenv("PERPLEXITY_API_URL", "https://api.perplexity.ai/chat/completions")
        try:
            self.max_tokens = int(os.getenv("PERPLEXITY_MAX_TOKENS", "4000"))
        except Exception:
            self.max_tokens = 4000

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def get_fundamentals(
        self,
        symbol: str,
        exchange: str,
        company_name: Optional[str] = None,
        years: int = 10,
        metrics: Optional[List[str]] = None,
        currency_hint: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Request EPS and ROE fundamentals for up to N fiscal years.

        Returns a dict:
        {
          "currency": "SGD",
          "eps_unit": "dollars" | "cents",
          "provenance": "perplexity:web",
          "data": {
             "2016": {"eps_basic": 0.25, "eps_diluted": 0.24, "roe_percent": 12.1, "eps_type": "basic", "source_urls": ["..."], "confidence": 0.78}
          }
        }
        """
        if not self.is_configured():
            logger.warning("Perplexity API key not configured; skipping Perplexity fetch")
            return None

        requested_metrics = metrics or ["eps_basic", "eps_diluted", "roe"]
        sys_prompt = (
            "You are a financial data assistant. Respond with STRICT JSON only (no commentary). "
            "Return EXACTLY N fiscal-year records (most recent completed ANNUAL periods; NO TTM or quarterly). "
            "For each year, provide basic EPS (not diluted) and ROE. eps_type MUST be 'basic'. Do not return diluted EPS. "
            "Prefer primary sources: company annual reports, financial statements, or investor relations pages. "
            "If a primary source is unavailable for a year, use reputable aggregators (e.g., Morningstar, WSJ, Macrotrends, StockAnalysis) with clear citation URLs. Do not use wikis or forums. "
            "For EVERY returned year, eps_basic and roe_percent MUST be numeric and non-null. If EPS is reported in cents, convert to dollars once. Round EPS to 3 decimals. "
            "Always include: currency code, eps_unit ('dollars' or 'cents'), per-year eps_type, and citation URLs for EACH year. "
            "If after exhaustive search basic EPS for any year truly cannot be found, include root fields 'error' and 'missing_years' listing those years."
        )

        # Build ticker variants to improve retrieval
        symbol_variants: List[str] = []
        try:
            if symbol:
                symbol_variants.append(symbol)
                if exchange:
                    symbol_variants.append(f"{symbol}.{exchange}")
                # Common SGX suffix used by many data providers
                symbol_variants.append(f"{symbol}.SI")
        except Exception:
            pass

        user_payload = {
            "symbol": symbol,
            "exchange": exchange,
            "company_name": company_name or "",
            "years": years,
            "metrics": requested_metrics,
            "currency_hint": currency_hint or "",
            "ticker_variants": symbol_variants,
            "requirements": {
                "coverage": "exactly N most recent fiscal years",
                "eps_policy": "basic EPS only; do not use diluted",
                "no_ttm": True,
                "citations_per_year": True,
                "prefer_primary_sources": True,
                "allow_reputable_aggregators": True,
                "aggregator_examples": ["morningstar", "wsj", "macrotrends", "stockanalysis"],
                "allow_derived_from_financials": True
            },
            "output_schema": {
                "currency": "string",
                "eps_unit": "dollars|cents",
                "provenance": "perplexity:web",
                "data": {
                    "<year>": {
                        "eps_basic": "number",
                        "roe_percent": "number",
                        "eps_type": "basic",
                        "source_urls": ["string"],
                        "confidence": "number between 0 and 1"
                    }
                }
            }
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        enable_search_classifier = os.getenv("PERPLEXITY_ENABLE_SEARCH_CLASSIFIER", "true").lower() == "true"
        # Determine if model is SONAR family to avoid response_format (prevents 400 errors)
        model_env_val = (self.model_name or "").strip().lower()
        is_sonar = "sonar" in model_env_val
        # Model auto-selection: if PERPLEXITY_MODEL == 'auto', try a prioritized list
        candidate_models: List[str] = []
        env_model = (self.model_name or "").strip().lower()
        if env_model == "auto":
            candidate_models = [
                "pplx-70b-online",
                "pplx-7b-online",
                "sonar-mini",
                "sonar-pro",
            ]
        else:
            candidate_models = [self.model_name]

        base_payload = {
            "model": candidate_models[0],
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "enable_search_classifier": enable_search_classifier,
        }

        def _post_and_parse(payload_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            resp = requests.post(self.api_url, headers=headers, json=payload_obj, timeout=self.timeout_seconds)
            status_code = resp.status_code
            text_body = resp.text
            try:
                data = resp.json()
            except Exception:
                data = None
            # Save raw response for debugging
            try:
                with open('perplexity_raw_response.json', 'w', encoding='utf-8') as f_raw:
                    json.dump({'status': status_code, 'body': data if data is not None else text_body}, f_raw, indent=2)
            except Exception:
                pass

            resp.raise_for_status()
            if data is None:
                logger.warning("Perplexity returned non-JSON body")
                return None

            content = self._extract_message_content(data)
            parsed = self._safe_json_parse(content)
            if not isinstance(parsed, dict) or "data" not in parsed:
                logger.warning("Perplexity returned unexpected structure; skipping")
                try:
                    with open('perplexity_parsed_content.txt', 'w', encoding='utf-8') as f_txt:
                        f_txt.write(content)
                except Exception:
                    pass
                return None

            # Normalize keys and values
            return self._normalize_response(parsed)

        try:
            # Try candidate models until we get enough basic EPS coverage or exhaust list
            best_result = None
            for idx, model_name_try in enumerate(candidate_models):
                payload = dict(base_payload)
                payload["model"] = model_name_try
                # For SONAR models, do not set response_format
                if "sonar" not in model_name_try.lower():
                    # For non-SONAR, we can request JSON schema strictly
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "fundamentals_payload",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "currency": {"type": "string"},
                                    "eps_unit": {"type": "string"},
                                    "provenance": {"type": "string"},
                                    "data": {
                                        "type": "object",
                                        "patternProperties": {
                                            "^[0-9]{4}$": {
                                                "type": "object",
                                                "properties": {
                                                    "eps_basic": {"type": ["number"]},
                                                    "eps_diluted": {"type": ["number", "null"]},
                                                    "roe_percent": {"type": ["number", "null"]},
                                                    "eps_type": {"type": ["string"], "enum": ["basic"]},
                                                    "source_urls": {"type": "array", "items": {"type": "string"}},
                                                    "confidence": {"type": "number"}
                                                },
                                                "required": ["eps_basic", "eps_type", "source_urls", "confidence"],
                                                "additionalProperties": False
                                            }
                                        },
                                        "additionalProperties": False
                                    }
                                },
                                "required": ["currency", "eps_unit", "provenance", "data"],
                                "additionalProperties": False
                            }
                        }
                    }
                result = _post_and_parse(payload)
                if result is None:
                    continue
                eps_years = [y for y, d in result.get("data", {}).items() if isinstance(d, dict) and d.get("basic_eps") is not None]
                if len(eps_years) >= years:
                    return result
                if best_result is None:
                    best_result = result
            return best_result
        except Exception as exc:
            logger.error(f"Perplexity request failed: {exc}")
            try:
                with open('perplexity_last_error.txt', 'w', encoding='utf-8') as f_err:
                    f_err.write(str(exc))
            except Exception:
                pass
            return None

    def get_pe_history(
        self,
        symbol: str,
        exchange: str,
        years: int = 10,
        company_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch per-year PE ratio history for up to N fiscal years via Perplexity.
        Returns dict: {"data": {"2016": {"pe_ratio": 15.2, "source_urls": ["..."], "confidence": 0.8}}, "provenance": "perplexity:web"}
        """
        if not self.is_configured():
            logger.warning("Perplexity API key not configured; skipping PE fetch")
            return None

        sys_prompt = (
            "You are a financial data assistant. Respond with STRICT JSON only (no commentary). "
            "Return EXACTLY N fiscal-year records (most recent completed ANNUAL periods; NO TTM or quarterly). "
            "For each year, provide the company's Price-to-Earnings (PE) ratio as a numeric value and include citation URLs. "
            "Prefer primary sources (company reports) or reputable aggregators (Morningstar, WSJ, Macrotrends, StockAnalysis). "
            "Always include: per-year pe_ratio (number), source_urls (array), and a confidence score between 0 and 1. "
            "If data for some year truly cannot be found, include root fields 'error' and 'missing_years'."
        )

        # Ticker variants to improve retrieval
        symbol_variants: List[str] = []
        try:
            if symbol:
                symbol_variants.append(symbol)
                if exchange:
                    symbol_variants.append(f"{symbol}.{exchange}")
                symbol_variants.append(f"{symbol}.SI")
        except Exception:
            pass

        user_payload = {
            "symbol": symbol,
            "exchange": exchange,
            "company_name": company_name or "",
            "years": years,
            "metric": "pe_ratio",
            "ticker_variants": symbol_variants,
            "requirements": {
                "coverage": "exactly N most recent fiscal years",
                "citations_per_year": True,
                "prefer_primary_sources": True,
                "allow_reputable_aggregators": True,
                "aggregator_examples": ["morningstar", "wsj", "macrotrends", "stockanalysis"]
            },
            "output_schema": {
                "provenance": "perplexity:web",
                "data": {
                    "<year>": {
                        "pe_ratio": "number",
                        "source_urls": ["string"],
                        "confidence": "number between 0 and 1"
                    }
                }
            }
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        enable_search_classifier = os.getenv("PERPLEXITY_ENABLE_SEARCH_CLASSIFIER", "true").lower() == "true"
        model_env_val = (self.model_name or "").strip().lower()
        is_sonar = "sonar" in model_env_val

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "enable_search_classifier": enable_search_classifier,
        }
        # Do not set response_format for SONAR models
        if not is_sonar:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "pe_history_payload",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "provenance": {"type": "string"},
                            "data": {
                                "type": "object",
                                "patternProperties": {
                                    "^[0-9]{4}$": {
                                        "type": "object",
                                        "properties": {
                                            "pe_ratio": {"type": ["number"]},
                                            "source_urls": {"type": "array", "items": {"type": "string"}},
                                            "confidence": {"type": "number"}
                                        },
                                        "required": ["pe_ratio", "source_urls", "confidence"],
                                        "additionalProperties": False
                                    }
                                },
                                "additionalProperties": False
                            }
                        },
                        "required": ["provenance", "data"],
                        "additionalProperties": False
                    }
                }
            }

        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout_seconds)
            resp.raise_for_status()
            data = resp.json()
            content = self._extract_message_content(data)
            parsed = self._safe_json_parse(content)
            if not isinstance(parsed, dict) or "data" not in parsed:
                return None
            # Normalize
            result_map: Dict[str, Any] = {}
            for y, obj in parsed.get("data", {}).items():
                try:
                    year = str(int(y))
                except Exception:
                    continue
                try:
                    pe_val = float(obj.get("pe_ratio")) if obj.get("pe_ratio") is not None else None
                except Exception:
                    pe_val = None
                result_map[year] = {
                    "pe_ratio": pe_val,
                    "source_urls": [str(s) for s in (obj.get("source_urls") or [])],
                    "confidence": float(obj.get("confidence")) if obj.get("confidence") is not None else 0.6
                }
            return {"provenance": "perplexity:web", "data": result_map}
        except Exception as exc:
            logger.error(f"Perplexity PE request failed: {exc}")
            return None

    def _extract_message_content(self, raw: Dict[str, Any]) -> str:
        try:
            choices = raw.get("choices", [])
            if not choices:
                return "{}"
            message = choices[0].get("message", {})
            content = message.get("content", "{}")
            return content if isinstance(content, str) else json.dumps(content)
        except Exception:
            return "{}"

    def _safe_json_parse(self, text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except Exception:
                    return {}
            return {}

    def _normalize_response(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        currency = payload.get("currency") or ""
        eps_unit = payload.get("eps_unit") or ""
        data_map = payload.get("data", {})

        needs_cent_to_dollar = (eps_unit.lower() == "cents")

        normalized_years: Dict[str, Any] = {}
        for year_str, year_obj in data_map.items():
            try:
                year = str(int(year_str))
            except Exception:
                continue

            eps_basic = year_obj.get("eps_basic")
            eps_diluted = year_obj.get("eps_diluted")
            roe_percent = year_obj.get("roe_percent")
            eps_type = year_obj.get("eps_type")
            sources = year_obj.get("source_urls", []) or []
            confidence = year_obj.get("confidence")

            def _conv(v: Optional[float]) -> Optional[float]:
                if v is None:
                    return None
                try:
                    fv = float(v)
                    return fv / 100.0 if needs_cent_to_dollar else fv
                except Exception:
                    return None

            normalized_years[year] = {
                "basic_eps": _conv(eps_basic) if eps_type == "basic" or eps_type is None else None,
                "diluted_eps": _conv(eps_diluted) if eps_type == "diluted" else None,
                "basic_roe": float(roe_percent) if roe_percent is not None else None,
                "source": "perplexity_api",
                "citations": [str(s) for s in sources],
                "confidence": float(confidence) if confidence is not None else 0.6,
            }

        return {
            "currency": currency,
            "eps_unit": "dollars" if needs_cent_to_dollar else (eps_unit or ""),
            "provenance": "perplexity:web",
            "data": normalized_years,
        }


__all__ = ["PerplexityClient"]


