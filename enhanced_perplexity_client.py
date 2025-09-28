#!/usr/bin/env python3
"""
Enhanced Perplexity Client with Hybrid Approach

Combines the reliability of our existing chat implementation with the new Search API capabilities.
Uses official Perplexity SDK for new features while maintaining backward compatibility.

Notes:
- ASCII-only strings and comments
- Hybrid approach: SDK for search + existing requests for chat
- Enhanced financial data retrieval with sub-document precision
- Real-time indexing and improved source discovery
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    from perplexity import Perplexity
    from perplexity._exceptions import APIError, APIConnectionError
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    Perplexity = None
    APIError = Exception
    APIConnectionError = Exception

logger = logging.getLogger(__name__)


class EnhancedPerplexityClient:
    """
    Hybrid Perplexity client combining SDK features with existing reliability.
    
    Features:
    - Search API for enhanced data discovery (via SDK)
    - Chat completions for structured data extraction (existing implementation)
    - Fallback mechanisms for API failures
    - Enhanced financial data retrieval
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, timeout_seconds: int = 45):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.model_name = model_name or os.getenv("PERPLEXITY_MODEL", "sonar-pro")
        self.timeout_seconds = timeout_seconds
        self.api_url = os.getenv("PERPLEXITY_API_URL", "https://api.perplexity.ai/chat/completions")
        
        try:
            self.max_tokens = int(os.getenv("PERPLEXITY_MAX_TOKENS", "4000"))
        except Exception:
            self.max_tokens = 4000
            
        # Initialize SDK client if available
        self.sdk_client = None
        if SDK_AVAILABLE and self.api_key:
            try:
                self.sdk_client = Perplexity(api_key=self.api_key)
                logger.info("Enhanced Perplexity client initialized with SDK support")
            except Exception as e:
                logger.warning(f"Failed to initialize SDK client: {e}")
                self.sdk_client = None
        
        # Fallback to existing implementation
        if not self.sdk_client:
            logger.info("Using fallback implementation (SDK not available or failed)")
    
    def is_configured(self) -> bool:
        """Check if client is properly configured"""
        return bool(self.api_key)
    
    def is_sdk_available(self) -> bool:
        """Check if SDK is available and working"""
        return self.sdk_client is not None
    
    def search_financial_sources(self, 
                                symbol: str, 
                                company_name: str, 
                                years: int = 10,
                                max_results: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        Use new Search API to discover comprehensive financial data sources.
        
        Returns list of search results with enhanced metadata for financial data.
        """
        if not self.is_sdk_available():
            logger.warning("Search API not available - SDK client not initialized")
            return None
        
        try:
            # Build comprehensive search query for financial data
            search_queries = [
                f"{company_name} {symbol} annual report financial statements",
                f"{company_name} {symbol} EPS earnings per share historical data",
                f"{company_name} {symbol} ROE return on equity financial metrics",
                f"{company_name} {symbol} investor relations financial results"
            ]
            
            all_results = []
            
            for query in search_queries:
                try:
                    search_response = self.sdk_client.search.create(
                        query=query,
                        max_results=max_results // len(search_queries) + 1
                    )
                    
                    # Process search results
                    for result in search_response.results:
                        result_data = {
                            "title": getattr(result, 'title', ''),
                            "url": getattr(result, 'url', ''),
                            "snippet": getattr(result, 'snippet', ''),
                            "query": query,
                            "relevance_score": getattr(result, 'relevance_score', 0.5)
                        }
                        all_results.append(result_data)
                        
                except Exception as e:
                    logger.warning(f"Search query failed: {query} - {e}")
                    continue
            
            # Remove duplicates and sort by relevance
            unique_results = []
            seen_urls = set()
            
            for result in all_results:
                if result['url'] not in seen_urls:
                    seen_urls.add(result['url'])
                    unique_results.append(result)
            
            # Sort by relevance score (if available)
            unique_results.sort(key=lambda x: x.get('relevance_score', 0.5), reverse=True)
            
            logger.info(f"Found {len(unique_results)} unique financial data sources")
            return unique_results[:max_results]
            
        except APIError as e:
            logger.error(f"Search API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in search: {e}")
            return None
    
    def get_fundamentals_enhanced(self,
                                 symbol: str,
                                 exchange: str,
                                 company_name: Optional[str] = None,
                                 years: int = 10,
                                 metrics: Optional[List[str]] = None,
                                 currency_hint: Optional[str] = None,
                                 use_search_api: bool = True) -> Optional[Dict[str, Any]]:
        """
        Enhanced fundamentals retrieval using hybrid approach.
        
        Phase 1: Use Search API to discover comprehensive sources
        Phase 2: Use existing chat implementation for structured extraction
        """
        if not self.is_configured():
            logger.warning("Perplexity API key not configured")
            return None
        
        company_name = company_name or symbol
        
        # Phase 1: Discover sources using Search API
        discovered_sources = []
        if use_search_api and self.is_sdk_available():
            logger.info("Phase 1: Discovering financial data sources via Search API")
            discovered_sources = self.search_financial_sources(symbol, company_name, years)
            
            if discovered_sources:
                logger.info(f"Discovered {len(discovered_sources)} potential sources")
                # Save discovered sources for debugging
                try:
                    with open('discovered_financial_sources.json', 'w', encoding='utf-8') as f:
                        json.dump(discovered_sources, f, indent=2)
                except Exception:
                    pass
            else:
                logger.warning("No sources discovered via Search API, falling back to chat-only")
        
        # Phase 2: Extract structured data using existing reliable chat implementation
        logger.info("Phase 2: Extracting structured financial data via Chat API")
        return self._get_fundamentals_via_chat(symbol, exchange, company_name, years, metrics, currency_hint, discovered_sources)
    
    def _get_fundamentals_via_chat(self,
                                  symbol: str,
                                  exchange: str,
                                  company_name: str,
                                  years: int,
                                  metrics: Optional[List[str]],
                                  currency_hint: Optional[str],
                                  discovered_sources: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Use existing reliable chat implementation for structured data extraction.
        Enhanced with discovered sources context.
        """
        requested_metrics = metrics or ["eps_basic", "eps_diluted", "roe"]
        
        # Enhanced system prompt with discovered sources context
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
        
        # Build ticker variants
        symbol_variants: List[str] = []
        try:
            if symbol:
                symbol_variants.append(symbol)
                if exchange:
                    symbol_variants.append(f"{symbol}.{exchange}")
                symbol_variants.append(f"{symbol}.SI")
        except Exception:
            pass
        
        # Enhanced user payload with discovered sources
        user_payload = {
            "symbol": symbol,
            "exchange": exchange,
            "company_name": company_name,
            "years": years,
            "metrics": requested_metrics,
            "currency_hint": currency_hint or "",
            "ticker_variants": symbol_variants,
            "discovered_sources": discovered_sources[:5] if discovered_sources else [],  # Include top 5 sources
            "requirements": {
                "coverage": "exactly N most recent fiscal years",
                "eps_policy": "basic EPS only; do not use diluted",
                "no_ttm": True,
                "citations_per_year": True,
                "prefer_primary_sources": True,
                "allow_reputable_aggregators": True,
                "aggregator_examples": ["morningstar", "wsj", "macrotrends", "stockanalysis"],
                "allow_derived_from_financials": True,
                "use_discovered_sources": bool(discovered_sources)
            },
            "output_schema": {
                "currency": "string",
                "eps_unit": "dollars|cents",
                "provenance": "perplexity:enhanced",
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
        
        # Model selection with fallback
        candidate_models = [self.model_name]
        if self.model_name.lower() == "auto":
            candidate_models = [
                "pplx-70b-online",
                "pplx-7b-online", 
                "sonar-mini",
                "sonar-pro",
            ]
        
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
            try:
                resp = requests.post(self.api_url, headers=headers, json=payload_obj, timeout=self.timeout_seconds)
                resp.raise_for_status()
                data = resp.json()
                
                content = self._extract_message_content(data)
                parsed = self._safe_json_parse(content)
                
                if not isinstance(parsed, dict) or "data" not in parsed:
                    logger.warning("Perplexity returned unexpected structure")
                    return None
                
                return self._normalize_response(parsed)
                
            except Exception as e:
                logger.error(f"Chat API request failed: {e}")
                return None
        
        try:
            # Try candidate models
            best_result = None
            for model_name_try in candidate_models:
                payload = dict(base_payload)
                payload["model"] = model_name_try
                
                # Add JSON schema for non-SONAR models
                if "sonar" not in model_name_try.lower():
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "enhanced_fundamentals_payload",
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
                    
                eps_years = [y for y, d in result.get("data", {}).items() 
                           if isinstance(d, dict) and d.get("basic_eps") is not None]
                
                if len(eps_years) >= years:
                    return result
                    
                if best_result is None:
                    best_result = result
            
            return best_result
            
        except Exception as e:
            logger.error(f"Enhanced fundamentals request failed: {e}")
            return None
    
    def _extract_message_content(self, raw: Dict[str, Any]) -> str:
        """Extract message content from API response"""
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
        """Safely parse JSON from response text"""
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
        """Normalize response to standard format"""
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
                "basic_roe": float(roe_percent) if roe_percent is not None else None,
                "source": "perplexity_enhanced",
                "citations": [str(s) for s in sources],
                "confidence": float(confidence) if confidence is not None else 0.6,
            }
        
        return {
            "currency": currency,
            "eps_unit": "dollars" if needs_cent_to_dollar else (eps_unit or ""),
            "provenance": "perplexity:enhanced",
            "data": normalized_years,
        }
    
    def get_enhanced_capabilities(self) -> Dict[str, Any]:
        """Return information about enhanced capabilities"""
        return {
            "sdk_available": self.is_sdk_available(),
            "search_api_enabled": self.is_sdk_available(),
            "chat_api_enabled": True,
            "hybrid_mode": True,
            "enhanced_features": [
                "Search API for source discovery",
                "Sub-document precision",
                "Real-time indexing",
                "Enhanced error handling",
                "Fallback mechanisms"
            ]
        }


__all__ = ["EnhancedPerplexityClient"]
