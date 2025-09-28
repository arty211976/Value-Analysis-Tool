#!/usr/bin/env python3
"""
Perplexity MCP Client

Fetches 10-year EPS/ROE fundamentals via Perplexity API, returning a
standardized JSON structure suitable for reconciliation with extracted
annual report data.

Notes:
- This module uses ASCII-only strings and comments.
- Perplexity output is constrained to strict JSON by prompt.
- EPS units and types are preserved; cents are converted to dollars once.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
import requests


logger = logging.getLogger(__name__)


class PerplexityMCPClient:
    """Thin wrapper around Perplexity chat completions for fundamentals."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, timeout_seconds: int = 45):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.model_name = model_name or os.getenv("PERPLEXITY_MODEL", "sonar-reasoning")
        self.timeout_seconds = timeout_seconds
        self.api_url = os.getenv("PERPLEXITY_API_URL", "https://api.perplexity.ai/chat/completions")
        try:
            self.max_tokens = int(os.getenv("PERPLEXITY_MAX_TOKENS", "3200"))
        except Exception:
            self.max_tokens = 3200

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
            logger.warning("Perplexity API key not configured; skipping MCP fetch")
            return None

        requested_metrics = metrics or ["eps_basic", "eps_diluted", "roe"]
        sys_prompt = (
            "You are a financial data assistant. Respond with STRICT JSON only. "
            "Do not include any commentary. Provide per-year fundamentals for the last N fiscal years. "
            "Always include: currency code, eps_unit (dollars or cents), and per-year EPS type (basic or diluted) if applicable. "
            "Include citation URLs for each year. If unsure, set the field to null and include an explanation note field at root."
        )

        user_prompt = {
            "symbol": symbol,
            "exchange": exchange,
            "company_name": company_name or "",
            "years": years,
            "metrics": requested_metrics,
            "currency_hint": currency_hint or "",
            "output_schema": {
                "currency": "string",
                "eps_unit": "dollars|cents",
                "provenance": "perplexity:web",
                "data": {
                    "<year>": {
                        "eps_basic": "number|null",
                        "eps_diluted": "number|null",
                        "roe_percent": "number|null",
                        "eps_type": "basic|diluted|null",
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
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            # Enforce strict JSON shape via Perplexity response_format JSON schema
            "response_format": {
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
                                            "eps_basic": {"type": ["number", "null"]},
                                            "eps_diluted": {"type": ["number", "null"]},
                                            "roe_percent": {"type": ["number", "null"]},
                                            "eps_type": {"type": ["string", "null"]},
                                            "source_urls": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "confidence": {"type": "number"}
                                        },
                                        "required": ["source_urls", "confidence"],
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
            },
        }

        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=self.timeout_seconds)
            status_code = resp.status_code
            text_body = resp.text
            try:
                data = resp.json()
            except Exception:
                data = None
            # Save raw response for debugging
            try:
                with open('mcp_perplexity_raw_response.json', 'w', encoding='utf-8') as f_raw:
                    json.dump({
                        'status': status_code,
                        'body': data if data is not None else text_body
                    }, f_raw, indent=2)
            except Exception:
                pass

            resp.raise_for_status()
            if data is None:
                logger.warning("Perplexity MCP returned non-JSON body")
                return None

            content = self._extract_message_content(data)
            parsed = self._safe_json_parse(content)
            if not isinstance(parsed, dict) or "data" not in parsed:
                logger.warning("Perplexity MCP returned unexpected structure; skipping")
                # Save parsed content for inspection
                try:
                    with open('mcp_perplexity_parsed_content.txt', 'w', encoding='utf-8') as f_txt:
                        f_txt.write(content)
                except Exception:
                    pass
                return None

            # Normalize and enforce ASCII keys
            normalized = self._normalize_response(parsed)
            return normalized
        except Exception as exc:
            logger.error(f"Perplexity MCP request failed: {exc}")
            try:
                with open('mcp_perplexity_last_error.txt', 'w', encoding='utf-8') as f_err:
                    f_err.write(str(exc))
            except Exception:
                pass
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
            # Attempt to locate a JSON object in the text
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

        # Convert cents to dollars once if eps_unit indicates cents
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
                    if needs_cent_to_dollar:
                        return fv / 100.0
                    return fv
                except Exception:
                    return None

            normalized_years[year] = {
                "basic_eps": _conv(eps_basic) if eps_type == "basic" or eps_type is None else None,
                "diluted_eps": _conv(eps_diluted) if eps_type == "diluted" else None,
                "basic_roe": float(roe_percent) if roe_percent is not None else None,
                "source": "perplexity_mcp",
                "citations": [str(s) for s in sources],
                "confidence": float(confidence) if confidence is not None else 0.6,
            }

        return {
            "currency": currency,
            "eps_unit": "dollars" if needs_cent_to_dollar else (eps_unit or ""),
            "provenance": "perplexity:web",
            "data": normalized_years,
        }


__all__ = ["PerplexityMCPClient"]


