#!/usr/bin/env python3
"""
Perplexity Integration Layer

Provides seamless integration between the enhanced Perplexity client and existing system.
Maintains backward compatibility while enabling enhanced features.

Notes:
- ASCII-only strings and comments
- Drop-in replacement for existing PerplexityClient
- Automatic fallback to original implementation if enhanced client fails
- Environment variable control for enhanced features
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import existing clients with different names to avoid circular imports
from perplexity_client import PerplexityClient as OriginalPerplexityClient
from perplexity_mcp_client import PerplexityMCPClient as OriginalPerplexityMCPClient

logger = logging.getLogger(__name__)

# Try to import enhanced client
try:
    from enhanced_perplexity_client import EnhancedPerplexityClient
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    EnhancedPerplexityClient = None


class PerplexityIntegration:
    """
    Integration layer providing seamless access to enhanced Perplexity features.
    
    Features:
    - Automatic fallback to original implementation
    - Environment variable control
    - Enhanced capabilities when available
    - Backward compatibility
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, timeout_seconds: int = 45):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.model_name = model_name or os.getenv("PERPLEXITY_MODEL", "sonar-pro")
        self.timeout_seconds = timeout_seconds
        
        # Check if enhanced features are enabled
        self.use_enhanced = (
            ENHANCED_AVAILABLE and 
            os.getenv("PERPLEXITY_USE_ENHANCED", "true").lower() == "true"
        )
        
        # Initialize clients
        self.enhanced_client = None
        self.original_client = None
        self.mcp_client = None
        
        if self.use_enhanced and ENHANCED_AVAILABLE:
            try:
                self.enhanced_client = EnhancedPerplexityClient(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    timeout_seconds=self.timeout_seconds
                )
                logger.info("Enhanced Perplexity client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize enhanced client: {e}")
                self.use_enhanced = False
        
        # Always initialize fallback clients
        self.original_client = OriginalPerplexityClient(
            api_key=self.api_key,
            model_name=self.model_name,
            timeout_seconds=self.timeout_seconds
        )
        
        self.mcp_client = OriginalPerplexityMCPClient(
            api_key=self.api_key,
            model_name=self.model_name,
            timeout_seconds=self.timeout_seconds
        )
        
        logger.info(f"Perplexity integration initialized - Enhanced: {self.use_enhanced}")
    
    def is_configured(self) -> bool:
        """Check if any client is properly configured"""
        return bool(self.api_key)
    
    def get_fundamentals(self,
                        symbol: str,
                        exchange: str,
                        company_name: Optional[str] = None,
                        years: int = 10,
                        metrics: Optional[List[str]] = None,
                        currency_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get fundamentals with enhanced capabilities and automatic fallback.
        
        Priority:
        1. Enhanced client (if available and enabled)
        2. Original client (fallback)
        3. MCP client (secondary fallback)
        """
        if not self.is_configured():
            logger.warning("Perplexity API key not configured")
            return None
        
        # Try enhanced client first
        if self.use_enhanced and self.enhanced_client:
            try:
                logger.info("Using enhanced Perplexity client for fundamentals")
                result = self.enhanced_client.get_fundamentals_enhanced(
                    symbol=symbol,
                    exchange=exchange,
                    company_name=company_name,
                    years=years,
                    metrics=metrics,
                    currency_hint=currency_hint,
                    use_search_api=True  # Enable search API if available
                )
                
                if result:
                    logger.info("Enhanced client returned successful results")
                    return result
                else:
                    logger.warning("Enhanced client returned no results, trying fallback")
                    
            except Exception as e:
                logger.error(f"Enhanced client failed: {e}, trying fallback")
        
        # Fallback to original client
        try:
            logger.info("Using original Perplexity client (fallback)")
            result = self.original_client.get_fundamentals(
                symbol=symbol,
                exchange=exchange,
                company_name=company_name,
                years=years,
                metrics=metrics,
                currency_hint=currency_hint
            )
            
            if result:
                logger.info("Original client returned successful results")
                return result
                
        except Exception as e:
            logger.error(f"Original client failed: {e}, trying MCP fallback")
        
        # Final fallback to MCP client
        try:
            logger.info("Using MCP Perplexity client (final fallback)")
            result = self.mcp_client.get_fundamentals(
                symbol=symbol,
                exchange=exchange,
                company_name=company_name,
                years=years,
                metrics=metrics,
                currency_hint=currency_hint
            )
            
            if result:
                logger.info("MCP client returned successful results")
                return result
                
        except Exception as e:
            logger.error(f"MCP client also failed: {e}")
        
        logger.error("All Perplexity clients failed to return results")
        return None
    
    def get_pe_history(self,
                      symbol: str,
                      exchange: str,
                      years: int = 10,
                      company_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get PE ratio history with fallback support.
        """
        if not self.is_configured():
            logger.warning("Perplexity API key not configured")
            return None
        
        # Try original client first (has PE history method)
        try:
            result = self.original_client.get_pe_history(
                symbol=symbol,
                exchange=exchange,
                years=years,
                company_name=company_name
            )
            
            if result:
                return result
                
        except Exception as e:
            logger.error(f"PE history request failed: {e}")
        
        return None
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get information about available capabilities"""
        capabilities = {
            "enhanced_available": ENHANCED_AVAILABLE,
            "enhanced_enabled": self.use_enhanced,
            "original_client": self.original_client is not None,
            "mcp_client": self.mcp_client is not None,
            "configured": self.is_configured()
        }
        
        if self.enhanced_client:
            enhanced_caps = self.enhanced_client.get_enhanced_capabilities()
            capabilities.update(enhanced_caps)
        
        return capabilities
    
    def search_financial_sources(self, 
                               symbol: str, 
                               company_name: str, 
                               years: int = 10,
                               max_results: int = 20) -> Optional[List[Dict[str, Any]]]:
        """
        Search for financial data sources using enhanced client.
        Falls back gracefully if enhanced client not available.
        """
        if self.use_enhanced and self.enhanced_client:
            try:
                return self.enhanced_client.search_financial_sources(
                    symbol=symbol,
                    company_name=company_name,
                    years=years,
                    max_results=max_results
                )
            except Exception as e:
                logger.error(f"Enhanced search failed: {e}")
        
        logger.warning("Financial source search not available (enhanced client not enabled)")
        return None


# Backward compatibility: Create aliases for existing code
PerplexityClient = PerplexityIntegration  # Drop-in replacement
PerplexityMCPClient = PerplexityIntegration  # Drop-in replacement

__all__ = ["PerplexityIntegration", "PerplexityClient", "PerplexityMCPClient"]
