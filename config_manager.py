#!/usr/bin/env python3
"""
Configuration Manager for Dynamic Company Analysis
Handles loading, updating, and managing company-specific configurations
"""

import json
import os
import re
from datetime import datetime
from typing import Dict, Any, Optional
import logging

class ConfigManager:
    """
    Manages company-specific configuration for dynamic analysis
    """
    
    def __init__(self, config_file: str = "company_config.json"):
        """
        Initialize the configuration manager
        
        Args:
            config_file: Path to the JSON configuration file
        """
        self.config_file = config_file
        self.config = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Default company values - these will be used if no specific company is set
        self.default_company = {
            "name": "SATS",
            "ticker": "S58",
            "industry": "Aviation",
            "sector": "Aviation Services",
            "country": "Singapore",
            "currency": "SGD",
            "eps_unit": "cents",
            "competitor_1": "Swissport International",
            "competitor_2": "Menzies Aviation", 
            "competitor_3": "Dnata"
        }
        
        # Current company values - will be populated when set_company is called
        self.current_company = None
        
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from JSON file
        
        Returns:
            Dictionary containing the configuration
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.logger.info(f"Configuration loaded from {self.config_file}")
            else:
                self.logger.warning(f"Configuration file {self.config_file} not found. Creating default config.")
                self.create_default_config()
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self.create_default_config()
        
        return self.config
    
    def create_default_config(self):
        """Create a default configuration with template placeholders"""
        self.config = {
            "company": {
                "name": "{{COMPANY_NAME}}",
                "ticker": "{{TICKER_SYMBOL}}",
                "ticker_variants": ["{{TICKER_SYMBOL}}"],
                "industry": "{{INDUSTRY}}",
                "sector": "{{SECTOR}}",
                "country": "{{COUNTRY}}",
                "currency": "{{CURRENCY}}",
                "eps_unit": "{{EPS_UNIT}}"
            },
            "analysis_settings": {
                "eps_validation_strategy": "flag",
                "tolerance_percent": 20.0,
                "expected_ranges": {
                    "basic_eps": [-10, 30],
                    "pe_ratio": [10, 60]
                },
                "use_real_market_data": True,
                "save_market_data": True,
                "market_data_cache_hours": 24,
                "save_intermediate_data": True,
                "save_validated_data": True,
                "verbose_output": True
            },
            "market_data": {
                "yahoo_finance": {
                    "base_url": "https://query1.finance.yahoo.com/v8/finance/chart/",
                    "timeout": 10,
                    "max_retries": 3
                },
                "alpha_vantage": {
                    "base_url": "https://www.alphavantage.co/query",
                    "timeout": 10,
                    "max_retries": 3
                },
                "preferred_provider": "yahoo_finance"
            },
            "document_settings": {
                "filename_prefix": "{{COMPANY_NAME_LOWER}}_analysis_report",
                "include_swot_analysis": True,
                "include_market_analysis": True,
                "include_roe_analysis": True,
                "include_valuation_analysis": True
            },
            "ai_agents": {
                "market_researcher": {
                    "name": "Market Research Analyst",
                    "goal": "Research and analyze current market conditions, industry trends, and company-specific developments for {{COMPANY_NAME}}",
                    "backstory": "You are an expert market research analyst specializing in {{INDUSTRY}} sector. You have deep knowledge of market dynamics, competitive landscape, and industry trends.",
                    "tools": ["web_search", "company_research", "market_analysis"],
                    "tasks": [
                        "Research recent company developments and strategic initiatives",
                        "Analyze industry trends and market dynamics",
                        "Identify key competitors and competitive advantages",
                        "Assess regulatory environment and compliance requirements",
                        "Evaluate technology trends and digital transformation",
                        "Project market outlook and growth prospects"
                    ]
                },
                "financial_analyst": {
                    "name": "Financial Analyst",
                    "goal": "Analyze financial performance, valuation metrics, and investment potential for {{COMPANY_NAME}}",
                    "backstory": "You are a senior financial analyst with expertise in {{INDUSTRY}} sector valuation and investment analysis.",
                    "tools": ["financial_analysis", "valuation_tools", "market_data"],
                    "tasks": [
                        "Analyze historical financial performance",
                        "Evaluate current valuation metrics",
                        "Assess investment risks and opportunities",
                        "Project future financial performance",
                        "Provide investment recommendations"
                    ]
                },
                "industry_expert": {
                    "name": "Industry Expert",
                    "goal": "Provide deep industry insights and sector-specific analysis for {{COMPANY_NAME}}",
                    "backstory": "You are a recognized expert in the {{INDUSTRY}} sector with decades of experience analyzing market trends, competitive dynamics, and industry evolution.",
                    "tools": ["industry_research", "competitive_analysis", "regulatory_analysis"],
                    "tasks": [
                        "Analyze industry structure and competitive landscape",
                        "Evaluate regulatory environment and compliance trends",
                        "Assess technological disruption and innovation impact",
                        "Identify growth opportunities and market expansion potential",
                        "Evaluate sustainability and ESG considerations"
                    ]
                }
            },
            "dynamic_content_generation": {
                "enabled": True,
                "use_real_time_data": True,
                "include_news_analysis": True,
                "include_competitive_analysis": True,
                "include_regulatory_analysis": True,
                "include_technology_trends": True,
                "include_market_outlook": True,
                "content_sources": [
                    "real-time news feeds",
                    "financial databases",
                    "industry reports",
                    "regulatory filings",
                    "analyst reports",
                    "market data"
                ]
            }
        }
        
        self.save_config()
    
    def _update_config_with_company(self, company_info: Dict[str, Any]):
        """
        Update configuration with company-specific values by substituting templates
        
        Args:
            company_info: Dictionary containing company information
        """
        # Create a copy of the config to modify
        updated_config = json.loads(json.dumps(self.config))
        
        # Reset filename_prefix to template format if it's not already a template
        if 'document_settings' in updated_config:
            current_prefix = updated_config['document_settings'].get('filename_prefix', '')
            if not current_prefix.startswith('{{'):
                updated_config['document_settings']['filename_prefix'] = "{{COMPANY_NAME_LOWER}}_analysis_report"
        
        # Define template variables
        template_vars = {
            "{{COMPANY_NAME}}": company_info.get("name", ""),
            "{{TICKER_SYMBOL}}": company_info.get("ticker", ""),
            "{{INDUSTRY}}": company_info.get("industry", ""),
            "{{SECTOR}}": company_info.get("sector", ""),
            "{{COUNTRY}}": company_info.get("country", ""),
            "{{CURRENCY}}": company_info.get("currency", ""),
            "{{EPS_UNIT}}": company_info.get("eps_unit", ""),
            "{{COMPANY_NAME_LOWER}}": company_info.get("name", "").lower().replace(" ", "_")
        }
        
        # Convert config to string for template substitution
        config_str = json.dumps(updated_config, indent=2)
        
        # Perform template substitution
        for template, value in template_vars.items():
            config_str = config_str.replace(template, str(value))
        
        # Convert back to dictionary
        self.config = json.loads(config_str)
        
        # Store current company info
        self.current_company = company_info
        
        self.logger.info(f"Configuration updated for company: {company_info.get('name')}")
    
    def set_company(self, company_info: Dict[str, Any]):
        """
        Set the current company and update configuration with template substitution
        
        Args:
            company_info: Dictionary containing company information
        """
        self._update_config_with_company(company_info)
        self.save_config()
    
    def save_config(self):
        """Save configuration to JSON file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values
        
        Args:
            updates: Dictionary containing updates
        """
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        deep_update(self.config, updates)
        self.save_config()
    
    def get_company_info(self) -> Dict[str, Any]:
        """Get company information"""
        return self.config.get("company", {})
    
    def get_analysis_settings(self) -> Dict[str, Any]:
        """Get analysis settings"""
        return self.config.get("analysis_settings", {})
    
    def get_market_data_config(self) -> Dict[str, Any]:
        """Get market data configuration"""
        return self.config.get("market_data", {})
    
    def get_document_settings(self) -> Dict[str, Any]:
        """Get document settings"""
        return self.config.get("document_settings", {})
    
    def get_research_data(self) -> Dict[str, Any]:
        """Get research data"""
        return self.config.get("research_data", {})
    
    def get_crew_ai_settings(self) -> Dict[str, Any]:
        """Get CrewAI settings"""
        return self.config.get("crew_ai_settings", {})
    
    def update_company_info(self, company_info: Dict[str, Any]):
        """
        Update company information with a dictionary and trigger template substitution
        
        Args:
            company_info: Dictionary containing company information
        """
        # First update the company info
        updates = {"company": company_info}
        self.update_config(updates)
        
        # Then trigger template substitution to update filename_prefix and other template variables
        self._update_config_with_company(company_info)
    
    def update_company_info_legacy(self, company_name: str, ticker: str, industry: str = None, 
                           sector: str = None, country: str = None, currency: str = None):
        """
        Update company information (legacy method)
        
        Args:
            company_name: Name of the company
            ticker: Stock ticker symbol
            industry: Industry sector
            sector: Business sector
            country: Country of operation
            currency: Currency for financial data
        """
        company_info = {
            "name": company_name,
            "ticker": ticker,
            "ticker_variants": [ticker]
        }
        
        if industry:
            company_info["industry"] = industry
        if sector:
            company_info["sector"] = sector
        if country:
            company_info["country"] = country
        if currency:
            company_info["currency"] = currency
        
        # Use the new update_company_info method which handles template substitution
        self.update_company_info(company_info)
    
    def update_research_data(self, research_data: Dict[str, Any]):
        """
        Update research data
        
        Args:
            research_data: Dictionary containing research data
        """
        updates = {"research_data": research_data}
        self.update_config(updates)
    
    def update_analysis_settings(self, settings: Dict[str, Any]):
        """
        Update analysis settings
        
        Args:
            settings: Dictionary containing analysis settings
        """
        updates = {"analysis_settings": settings}
        self.update_config(updates)
    
    def get_filename_prefix(self) -> str:
        """Get the filename prefix for generated reports"""
        return self.config.get("document_settings", {}).get("filename_prefix", "analysis_report")
    
    def get_ticker(self) -> str:
        """Get the current ticker symbol"""
        return self.config.get("company", {}).get("ticker", "DEFAULT.SI")
    
    def get_company_name(self) -> str:
        """Get the current company name"""
        return self.config.get("company", {}).get("name", "Default Company")
    
    def get_industry(self) -> str:
        """Get the current industry"""
        return self.config.get("company", {}).get("industry", "general")
    
    def get_currency(self) -> str:
        """Get the current currency"""
        return self.config.get("company", {}).get("currency", "SGD")
    
    def get_eps_unit(self) -> str:
        """Get the EPS unit"""
        return self.config.get("company", {}).get("eps_unit", "cents")
    
    def print_config_summary(self):
        """Print a summary of the current configuration"""
        print("Current Configuration Summary")
        print("=" * 50)
        
        company = self.get_company_info()
        print(f"Company: {company.get('name', 'N/A')}")
        print(f"Ticker: {company.get('ticker', 'N/A')}")
        print(f"Industry: {company.get('industry', 'N/A')}")
        print(f"Country: {company.get('country', 'N/A')}")
        print(f"Currency: {company.get('currency', 'N/A')}")
        
        analysis_settings = self.get_analysis_settings()
        print(f"EPS Validation Strategy: {analysis_settings.get('eps_validation_strategy', 'N/A')}")
        print(f"Tolerance: {analysis_settings.get('tolerance_percent', 'N/A')}%")
        print(f"Use Real Market Data: {analysis_settings.get('use_real_market_data', 'N/A')}")
        
        document_settings = self.get_document_settings()
        print(f"Filename Prefix: {document_settings.get('filename_prefix', 'N/A')}")
        
        print("\nTo change configuration:")
        print("   1. Edit company_config.json directly")
        print("   2. Use ConfigManager methods programmatically")
        print("   3. Restart the analysis script")

def create_company_config(company_name: str, ticker: str, industry: str = None, 
                         sector: str = None, country: str = None, currency: str = None,
                         config_file: str = "company_config.json"):
    """
    Create a new company configuration
    
    Args:
        company_name: Name of the company
        ticker: Stock ticker symbol
        industry: Industry sector
        sector: Business sector
        country: Country of operation
        currency: Currency for financial data
        config_file: Path to save the configuration file
    
    Returns:
        ConfigManager instance with the new configuration
    """
    config_manager = ConfigManager(config_file)
    config_manager.update_company_info(company_name, ticker, industry, sector, country, currency)
    return config_manager

if __name__ == "__main__":
    # Example usage
    config_manager = ConfigManager()
    config_manager.print_config_summary() 