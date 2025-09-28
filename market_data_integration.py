#!/usr/bin/env python3
"""
Market Data Integration for P/E Ratios
Fetches P/E ratios and other market data from financial APIs
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataIntegration:
    """
    Market data integration for fetching P/E ratios and other market metrics.
    """
    
    def __init__(self):
        # Industry average P/E ratios as fallback
        self.industry_pe_ratios = {
            'Bank': 12.0,
            'Banking': 12.0,
            'Financial Services': 14.0,
            'Technology': 25.0,
            'Healthcare': 20.0,
            'Consumer Goods': 18.0,
            'Energy': 15.0,
            'Transportation': 16.0,
            'Manufacturing': 17.0,
            'Real Estate': 15.0,
            'Telecommunications': 16.0,
            'Utilities': 18.0,
            'Retail': 19.0,
            'Insurance': 13.0,
            'general': 18.0  # Default fallback
        }
        
        # API configuration
        self.api_keys = {
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'yahoo_finance': None,  # No API key needed
            'polygon': os.getenv('POLYGON_API_KEY')
        }
    
    def get_pe_ratio_from_alpha_vantage(self, ticker_symbol: str) -> Optional[float]:
        """
        Get P/E ratio from Alpha Vantage API.
        """
        if not self.api_keys['alpha_vantage']:
            logger.warning("Alpha Vantage API key not configured")
            return None
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'OVERVIEW',
                'symbol': ticker_symbol,
                'apikey': self.api_keys['alpha_vantage']
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'PERatio' in data and data['PERatio']:
                return float(data['PERRatio'])
            else:
                logger.info(f"No P/E ratio found for {ticker_symbol} in Alpha Vantage")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching P/E ratio from Alpha Vantage: {e}")
            return None
    
    def get_pe_ratio_from_yahoo_finance(self, ticker_symbol: str) -> Optional[float]:
        """
        Get P/E ratio from Yahoo Finance using yfinance library.
        """
        try:
            import yfinance as yf
            
            # Get stock info from Yahoo Finance
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            
            # Extract P/E ratio
            if 'trailingPE' in info and info['trailingPE'] is not None:
                pe_ratio = float(info['trailingPE'])
                logger.info(f"Found P/E ratio from Yahoo Finance for {ticker_symbol}: {pe_ratio}")
                return pe_ratio
            elif 'forwardPE' in info and info['forwardPE'] is not None:
                pe_ratio = float(info['forwardPE'])
                logger.info(f"Found forward P/E ratio from Yahoo Finance for {ticker_symbol}: {pe_ratio}")
                return pe_ratio
            else:
                logger.info(f"No P/E ratio found for {ticker_symbol} in Yahoo Finance")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching P/E ratio from Yahoo Finance: {e}")
            return None
    
    def get_pe_ratio_from_polygon(self, ticker_symbol: str) -> Optional[float]:
        """
        Get P/E ratio from Polygon API.
        """
        if not self.api_keys['polygon']:
            logger.warning("Polygon API key not configured")
            return None
        
        try:
            url = f"https://api.polygon.io/v3/reference/tickers/{ticker_symbol}"
            params = {
                'apiKey': self.api_keys['polygon']
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Polygon doesn't directly provide P/E ratio, but we can calculate it
            # from price and earnings data if available
            logger.info(f"Polygon integration requires additional implementation for P/E ratio")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching data from Polygon: {e}")
            return None
    
    def get_industry_pe_ratio(self, industry: str) -> float:
        """
        Get industry average P/E ratio as fallback.
        """
        industry_lower = industry.lower()
        
        # Find the best match for the industry
        for key, value in self.industry_pe_ratios.items():
            if key.lower() in industry_lower or industry_lower in key.lower():
                return value
        
        # Return default if no match found
        return self.industry_pe_ratios['general']
    
    def calculate_pe_ratio_from_eps_and_price(self, eps: float, stock_price: float) -> Optional[float]:
        """
        Calculate P/E ratio from EPS and stock price.
        """
        if eps and stock_price and eps > 0:
            return stock_price / eps
        return None
    
    def get_current_stock_price(self, ticker_symbol: str) -> Optional[float]:
        """
        Get current stock price from Yahoo Finance.
        """
        try:
            import yfinance as yf
            
            # Get stock info from Yahoo Finance
            stock = yf.Ticker(ticker_symbol)
            info = stock.info
            
            # Extract current price
            if 'currentPrice' in info and info['currentPrice'] is not None:
                current_price = float(info['currentPrice'])
                logger.info(f"Found current price from Yahoo Finance for {ticker_symbol}: {current_price}")
                return current_price
            elif 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                current_price = float(info['regularMarketPrice'])
                logger.info(f"Found market price from Yahoo Finance for {ticker_symbol}: {current_price}")
                return current_price
            else:
                logger.info(f"No current price found for {ticker_symbol} in Yahoo Finance")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching current price from Yahoo Finance: {e}")
            return None
    
    def get_market_data_for_analysis(self, ticker_symbol: str, industry: str = 'general') -> Dict[str, Any]:
        """
        Get comprehensive market data including P/E ratios.
        """
        logger.info(f"Fetching market data for {ticker_symbol}")
        
        # Try multiple sources for P/E ratio
        pe_ratio = None
        pe_source = None
        
        # Try Alpha Vantage first
        if self.api_keys['alpha_vantage']:
            pe_ratio = self.get_pe_ratio_from_alpha_vantage(ticker_symbol)
            if pe_ratio:
                pe_source = 'alpha_vantage'
        
        # Try Yahoo Finance if Alpha Vantage failed
        if not pe_ratio:
            pe_ratio = self.get_pe_ratio_from_yahoo_finance(ticker_symbol)
            if pe_ratio:
                pe_source = 'yahoo_finance'
        
        # Try Polygon if others failed
        if not pe_ratio:
            pe_ratio = self.get_pe_ratio_from_polygon(ticker_symbol)
            if pe_ratio:
                pe_source = 'polygon'
        
        # Use industry average as fallback
        if not pe_ratio:
            pe_ratio = self.get_industry_pe_ratio(industry)
            pe_source = 'industry_average'
            logger.info(f"Using industry average P/E ratio for {industry}: {pe_ratio}")
        
        # Get current stock price
        current_price = self.get_current_stock_price(ticker_symbol)
        
        return {
            "pe_ratio": pe_ratio,
            "pe_source": pe_source,
            "current_price": current_price,
            "ticker_symbol": ticker_symbol,
            "industry": industry,
            "timestamp": datetime.now().isoformat()
        }
    
    def enhance_eps_data_with_market_data(self, eps_data: Dict, ticker_symbol: str, industry: str = 'general') -> Dict:
        """
        Enhance EPS data with market-derived P/E ratios.
        """
        logger.info(f"Enhancing EPS data with market data for {ticker_symbol}")
        
        # Get market data
        market_data = self.get_market_data_for_analysis(ticker_symbol, industry)
        
        # Enhance EPS data with P/E ratio
        enhanced_eps_data = eps_data.copy()
        
        # Add P/E ratio to the most recent year with EPS data
        if market_data['pe_ratio']:
            # Find the most recent year with EPS data
            years_with_eps = [year for year, data in eps_data.items() 
                            if isinstance(data, dict) and data.get('basic_eps') is not None]
            
            if years_with_eps:
                most_recent_year = max(years_with_eps)
                if most_recent_year in enhanced_eps_data:
                    enhanced_eps_data[most_recent_year]['pe_ratio'] = market_data['pe_ratio']
                    enhanced_eps_data[most_recent_year]['pe_source'] = market_data['pe_source']
        
        # Add market data metadata
        enhanced_eps_data['market_data'] = market_data
        
        return enhanced_eps_data

def main():
    """Test the market data integration."""
    market_data = MarketDataIntegration()
    
    # Test with OCBC ticker
    ticker_symbol = "O39.SI"  # OCBC Bank Singapore
    industry = "Bank"
    
    print(f"Testing market data integration for {ticker_symbol}")
    print()
    
    # Get market data
    result = market_data.get_market_data_for_analysis(ticker_symbol, industry)
    
    print("=== MARKET DATA RESULTS ===")
    print(f"Ticker Symbol: {result['ticker_symbol']}")
    print(f"Industry: {result['industry']}")
    print(f"P/E Ratio: {result['pe_ratio']}")
    print(f"P/E Source: {result['pe_source']}")
    print(f"Current Price: {result['current_price']}")
    print(f"Timestamp: {result['timestamp']}")
    
    # Test with sample EPS data
    sample_eps_data = {
        "2024": {"basic_eps": 1.07},
        "2023": {"basic_eps": 0.80},
        "2022": {"basic_eps": 1.12}
    }
    
    print()
    print("=== ENHANCED EPS DATA TEST ===")
    enhanced_data = market_data.enhance_eps_data_with_market_data(
        sample_eps_data, ticker_symbol, industry
    )
    
    print("Enhanced EPS Data:")
    for year, data in enhanced_data.items():
        if isinstance(data, dict):
            print(f"  {year}: EPS={data.get('basic_eps')}, P/E={data.get('pe_ratio')}")

if __name__ == "__main__":
    main() 