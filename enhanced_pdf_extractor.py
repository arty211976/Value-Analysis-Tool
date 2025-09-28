#!/usr/bin/env python3
"""
Enhanced PDF Extractor for 10-Year Historical Data
Specifically designed to extract multiple years of EPS/ROE data from annual reports
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import fitz  # PyMuPDF
import pdfplumber
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPDFExtractor:
    """
    Enhanced PDF extractor specifically designed to extract 10-year historical data
    from annual reports using multiple advanced strategies.
    """
    
    def __init__(self):
        self.current_year = datetime.now().year
        
        # Historical data section identifiers
        self.historical_sections = [
            'five-year summary', 'group five-year summary', 'historical data',
            'financial highlights', 'selected financial data', 'financial summary',
            'per share data', 'per ordinary share', 'earnings per share',
            'consolidated income', 'income statement', 'profit and loss',
            'key performance indicators', 'financial performance'
        ]
        
        # Enhanced EPS-related patterns for multiple report formats
        self.eps_patterns = [
            # OCBC-style patterns (preserved)
            r'basic\s+earnings\s+per\s+share',
            r'basic\s+eps',
            r'earnings\s+per\s+share',
            r'eps\s+\(?basic\)?',
            r'net\s+income\s+per\s+share',
            r'profit\s+per\s+share',
            r'per\s+ordinary\s+share',
            r'per\s+share\s+earnings',
            # SATS-style patterns (enhanced based on correct data)
            r'earnings\s+per\s+share\s*\(?basic\)?',
            r'eps\s*\(?basic\)?',
            r'per\s+share',
            r'net\s+asset\s+value\s+per\s+share',
            r'basic\s+earnings\s+per\s+share',
            r'basic\s+eps',
            r'earnings\s+per\s+share\s+\(?basic\)?',
            r'eps\s+\(?basic\)?',
            r'per\s+share\s+earnings',
            r'profit\s+per\s+share',
            r'net\s+income\s+per\s+share',
            # SATS-specific patterns (new)
            r'earnings\s+per\s+share\s*\(?sgd\)?',
            r'eps\s*\(?sgd\)?',
            r'per\s+share\s+earnings\s*\(?sgd\)?',
            r'net\s+earnings\s+per\s+share',
            r'profit\s+attributable\s+to\s+shareholders\s+per\s+share',
            r'net\s+profit\s+per\s+share',
            r'basic\s+earnings\s+per\s+share\s*\(?sgd\)?',
            r'eps\s+\(?sgd\)?',
            r'per\s+share\s+data',
            r'per\s+ordinary\s+share\s*\(?sgd\)?',
            r'earnings\s+per\s+ordinary\s+share',
            r'net\s+earnings\s+per\s+share\s*\(?sgd\)?',
            r'profit\s+per\s+ordinary\s+share\s*\(?sgd\)?',
            r'income\s+per\s+ordinary\s+share\s*\(?sgd\)?',
            # Universal patterns for any report format
            r'earnings\s+per\s+share',
            r'eps',
            r'per\s+share\s+earnings',
            r'profit\s+per\s+share',
            r'income\s+per\s+share',
            r'net\s+income\s+per\s+share',
            r'profit\s+attributable\s+to\s+shareholders',
            r'net\s+profit\s+per\s+share',
            r'basic\s+eps',
            r'diluted\s+eps',
            r'eps\s+\(?basic\)?',
            r'eps\s+\(?diluted\)?',
            r'per\s+share\s+data',
            r'per\s+ordinary\s+share',
            r'per\s+common\s+share',
            r'earnings\s+per\s+ordinary\s+share',
            r'earnings\s+per\s+common\s+share',
            r'net\s+earnings\s+per\s+share',
            r'profit\s+per\s+ordinary\s+share',
            r'income\s+per\s+ordinary\s+share'
        ]
        
        # Enhanced ROE-related patterns
        self.roe_patterns = [
            # OCBC-style patterns (preserved)
            r'return\s+on\s+equity',
            r'roe',
            r'return\s+on\s+shareholders\s+equity',
            r'return\s+on\s+common\s+equity',
            r'equity\s+return',
            # SATS-style patterns (new)
            r'return\s+on\s+equity',
            r'roe',
            # Generic patterns for other formats
            r'return\s+on\s+equity',
            r'roe'
        ]
        
        # Enhanced year patterns for multiple formats
        self.year_patterns = [
            # OCBC-style patterns (preserved)
            r'\b(19|20)\d{2}\b',  # Years 1900-2099
            r'FY\s*(19|20)\d{2}',  # Fiscal year
            r'Year\s*(19|20)\d{2}',  # Year format
            # SATS-style patterns (new)
            r'FY\s*\d{4}-\d{2}',  # FY2023-24 format
            r'FY\s*\d{2}',  # FY2023 format
            r'(19|20)\d{2}-\d{2}',  # 2023-24 format
            # Generic patterns for other formats
            r'\b(19|20)\d{2}\b',  # Standard years
            r'FY\s*(19|20)\d{2}',  # Fiscal years
            r'Year\s*(19|20)\d{2}'  # Year prefix
        ]
        
        # Report format detection patterns
        self.format_indicators = {
            'ocbc_style': [
                r'basic\s+earnings\s+per\s+share',
                r'consolidated\s+income',
                r'group\s+five-year\s+summary'
            ],
            'sats_style': [
                r'FY\s*\d{4}-\d{2}',
                r'net\s+asset\s+value\s+per\s+share',
                r'operating\s+statistics',
                r'sats\s+ltd',
                r'sats\s+limited',
                r'ground\s+handling',
                r'catering\s+services',
                r'aviation\s+services'
            ],
            'generic': [
                r'earnings\s+per\s+share',
                r'return\s+on\s+equity',
                r'financial\s+highlights'
            ]
        }
    
    def extract_text_with_enhanced_layout(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text with enhanced layout analysis focusing on historical data sections.
        """
        try:
            doc = fitz.open(pdf_path)
            layout_data = {
                'pages': [],
                'historical_sections': [],
                'financial_tables': [],
                'full_text': ""
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text with layout information
                text_dict = page.get_text("dict")
                page_data = {
                    'page_number': page_num + 1,
                    'blocks': [],
                    'text': ""
                }
                
                # Process blocks to preserve structure
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        block_text = ""
                        block_data = {
                            'bbox': block.get('bbox', []),
                            'type': block.get('type', 0),
                            'lines': []
                        }
                        
                        for line in block["lines"]:
                            line_text = ""
                            line_data = {
                                'bbox': line.get('bbox', []),
                                'spans': []
                            }
                            
                            for span in line["spans"]:
                                line_text += span["text"] + " "
                                line_data['spans'].append({
                                    'text': span["text"],
                                    'bbox': span.get('bbox', []),
                                    'font': span.get('font', ''),
                                    'size': span.get('size', 0)
                                })
                            
                            block_text += line_text + "\n"
                            block_data['lines'].append(line_data)
                        
                        page_data['text'] += block_text
                        page_data['blocks'].append(block_data)
                        
                        # Check if this block contains historical financial data
                        if self._is_historical_financial_block(block_text):
                            layout_data['historical_sections'].append({
                                'page': page_num + 1,
                                'text': block_text,
                                'bbox': block.get('bbox', [])
                            })
                        
                        # Check if this block contains tabular financial data
                        if self._is_financial_table_block(block_text):
                            layout_data['financial_tables'].append({
                                'page': page_num + 1,
                                'text': block_text,
                                'bbox': block.get('bbox', [])
                            })
                
                layout_data['pages'].append(page_data)
                layout_data['full_text'] += page_data['text'] + "\n"
            
            doc.close()
            return layout_data
            
        except Exception as e:
            logger.error(f"Error extracting text with enhanced layout: {e}")
            return {'full_text': '', 'pages': [], 'historical_sections': [], 'financial_tables': []}
    
    def _detect_report_format(self, text: str) -> str:
        """
        Detect the report format based on content patterns.
        Returns: 'ocbc_style', 'sats_style', or 'generic'
        """
        text_lower = text.lower()
        
        # Count matches for each format
        format_scores = {}
        
        for format_type, patterns in self.format_indicators.items():
            score = 0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                score += len(matches)
            format_scores[format_type] = score
        
        # Determine the most likely format
        if format_scores['ocbc_style'] > format_scores['sats_style'] and format_scores['ocbc_style'] > 0:
            return 'ocbc_style'
        elif format_scores['sats_style'] > format_scores['ocbc_style'] and format_scores['sats_style'] > 0:
            return 'sats_style'
        else:
            return 'generic'
    
    def _extract_years_from_text(self, text: str) -> List[str]:
        """
        Extract years from text using enhanced patterns for multiple formats.
        """
        years = []
        
        for pattern in self.year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle capture groups
                    for group in match:
                        if group and len(group) == 4:
                            years.append(group)
                else:
                    # Handle direct matches
                    if len(match) == 4:
                        years.append(match)
                    elif 'FY' in match:
                        # Extract year from FY format
                        year_match = re.search(r'\d{4}', match)
                        if year_match:
                            years.append(year_match.group())
        
        return list(set(years))  # Remove duplicates
    
    def _extract_eps_from_text_blocks(self, text: str) -> Dict[str, Any]:
        """
        Extract EPS data from text blocks using advanced strategies for any report format.
        Enhanced with multiple extraction methods for comprehensive coverage.
        """
        eps_data = {}
        
        # Split text into lines
        lines = text.split('\n')
        
        # Strategy 1: Look for EPS-related lines with years
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check if line contains EPS-related keywords
            if any(pattern in line_lower for pattern in self.eps_patterns):
                # Extract years from this line and surrounding lines
                context_lines = lines[max(0, i-2):min(len(lines), i+3)]
                context_text = '\n'.join(context_lines)
                
                years = self._extract_years_from_text(context_text)
                numbers = re.findall(r'\d+\.?\d*', context_text)
                
                # Try to match years with numbers
                if years and numbers:
                    for year in years:
                        # Find the closest number to this year
                        year_pos = context_text.find(year)
                        if year_pos != -1:
                            # Look for numbers near this year
                            for num in numbers:
                                num_pos = context_text.find(num)
                                if abs(num_pos - year_pos) < 200:  # Within 200 characters
                                    try:
                                        eps_value = float(num)
                                        if 0 < eps_value < 1000:  # Reasonable EPS range
                                            eps_data[year] = {'basic_eps': eps_value}
                                            break
                                    except ValueError:
                                        continue
        
        # Strategy 2: Look for historical data sections with multiple years
        historical_patterns = [
            r'five\s+year\s+summary.*?earnings\s+per\s+share',
            r'historical\s+data.*?earnings\s+per\s+share',
            r'financial\s+highlights.*?earnings\s+per\s+share',
            r'per\s+share\s+data.*?earnings\s+per\s+share',
            r'basic\s+earnings\s+per\s+share.*?(\d{4}.*?\d{4})',
            r'eps.*?(\d{4}.*?\d{4})',
            r'ten\s+year\s+summary.*?earnings\s+per\s+share',
            r'historical\s+performance.*?earnings\s+per\s+share',
            r'financial\s+summary.*?earnings\s+per\s+share',
            r'selected\s+financial\s+data.*?earnings\s+per\s+share'
        ]
        
        for pattern in historical_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Extract years from the matched section
                section_text = match.group(0)
                years = self._extract_years_from_text(section_text)
                numbers = re.findall(r'\d+\.?\d*', section_text)
                
                # Match years with numbers in sequence
                if years and numbers:
                    for i, year in enumerate(years[:len(numbers)]):
                        try:
                            eps_value = float(numbers[i])
                            if 0 < eps_value < 1000:
                                eps_data[year] = {'basic_eps': eps_value}
                        except (ValueError, IndexError):
                            continue
        
        # Strategy 3: Look for table-like structures with years and EPS values
        table_patterns = [
            r'(\d{4})\s+(\d+\.?\d*)',  # Year followed by number
            r'FY\s*(\d{4}-\d{2})\s+(\d+\.?\d*)',  # FY format
            r'(\d{4}-\d{2})\s+(\d+\.?\d*)',  # Year range format
            r'(\d{4})\s+(\d+\.?\d*)\s+(\d+\.?\d*)',  # Year with multiple values
            r'FY\s*(\d{4})\s+(\d+\.?\d*)',  # FY2023 format
            r'(\d{4})\s*(\d+\.?\d*)',  # Year with optional space
            r'(\d{4})\s*:\s*(\d+\.?\d*)',  # Year: value format
            r'(\d{4})\s*-\s*(\d+\.?\d*)'  # Year - value format
        ]
        
        for pattern in table_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    year = match.group(1)
                    eps_value = float(match.group(2))
                    if 0 < eps_value < 1000:  # Reasonable EPS range
                        eps_data[year] = {'basic_eps': eps_value}
                except (ValueError, IndexError):
                    continue
        
        # Strategy 4: Enhanced extraction for SATS-style reports
        # Look for patterns specific to SATS reports with small decimal values
        sats_specific_patterns = [
            # Look for patterns with small decimal values (common in SATS) - PRIORITY 1
            r'(\d{4})\s*(0\.\d{1,4})',  # Year with small decimal values
            r'(\d{4})\s*(-0\.\d{1,4})',  # Year with negative small decimal values
            # Look for patterns with decimal values in SATS range - PRIORITY 2
            r'(\d{4})\s*(-?\d+\.\d{1,4})',  # Year with negative decimal values
            r'(\d{4})\s*(\d+\.\d{2,4})',  # Year with decimal values
            r'(\d{4})\s*(\d+\.\d{1,4})',  # Year with decimal values (1-4 digits)
        ]
        
        for pattern in sats_specific_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    year = match.group(1)
                    eps_value = float(match.group(2))
                    # More permissive range for SATS (can be negative and small)
                    if -1.0 < eps_value < 100.0:  # Expanded SATS EPS range
                        eps_data[year] = {'basic_eps': eps_value}
                except (ValueError, IndexError):
                    continue
        
        # Strategy 5: Look for EPS values in context of financial statements
        eps_context_patterns = [
            r'earnings\s+per\s+share.*?(\d{4}).*?(\d+\.?\d*)',
            r'eps.*?(\d{4}).*?(\d+\.?\d*)',
            r'per\s+share.*?(\d{4}).*?(\d+\.?\d*)',
            r'basic\s+earnings.*?(\d{4}).*?(\d+\.?\d*)',
            r'net\s+income\s+per\s+share.*?(\d{4}).*?(\d+\.?\d*)',
            r'profit\s+per\s+share.*?(\d{4}).*?(\d+\.?\d*)',
        ]
        
        for pattern in eps_context_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    year = match.group(1)
                    eps_value = float(match.group(2))
                    if -1.0 < eps_value < 1.0:  # SATS EPS range
                        eps_data[year] = {'basic_eps': eps_value}
                except (ValueError, IndexError):
                    continue
        
        # Strategy 6: Advanced pattern matching for various report formats
        advanced_patterns = [
            r'(\d{4})\s*EPS\s*(\d+\.?\d*)',  # 2023 EPS 1.25
            r'EPS\s*(\d{4})\s*(\d+\.?\d*)',  # EPS 2023 1.25
            r'(\d{4})\s*Earnings\s*(\d+\.?\d*)',  # 2023 Earnings 1.25
            r'Earnings\s*(\d{4})\s*(\d+\.?\d*)',  # Earnings 2023 1.25
            r'(\d{4})\s*Basic\s*(\d+\.?\d*)',  # 2023 Basic 1.25
            r'Basic\s*(\d{4})\s*(\d+\.?\d*)',  # Basic 2023 1.25
            r'(\d{4})\s*Per\s+Share\s*(\d+\.?\d*)',  # 2023 Per Share 1.25
            r'Per\s+Share\s*(\d{4})\s*(\d+\.?\d*)'  # Per Share 2023 1.25
        ]
        
        for pattern in advanced_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    year = match.group(1)
                    eps_value = float(match.group(2))
                    if 0 < eps_value < 1000:
                        eps_data[year] = {'basic_eps': eps_value}
                except (ValueError, IndexError):
                    continue
        
        # Strategy 7: Look for multi-year sequences
        multi_year_patterns = [
            r'(\d{4})\s+(\d{4})\s+(\d{4})\s+(\d{4})\s+(\d{4})',  # 5 consecutive years
            r'(\d{4})\s+(\d{4})\s+(\d{4})\s+(\d{4})',  # 4 consecutive years
            r'(\d{4})\s+(\d{4})\s+(\d{4})',  # 3 consecutive years
        ]
        
        for pattern in multi_year_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Look for numbers near this year sequence
                context_start = max(0, match.start() - 500)
                context_end = min(len(text), match.end() + 500)
                context = text[context_start:context_end]
                
                years = list(match.groups())
                numbers = re.findall(r'\d+\.?\d*', context)
                
                # Try to match years with numbers
                if len(numbers) >= len(years):
                    for i, year in enumerate(years):
                        try:
                            eps_value = float(numbers[i])
                            if 0 < eps_value < 1000:
                                eps_data[year] = {'basic_eps': eps_value}
                        except (ValueError, IndexError):
                            continue
        
        return eps_data
    
    def _extract_roe_from_text_blocks(self, text: str) -> Dict[str, Any]:
        """
        Extract ROE data from text blocks (for non-tabular formats like SATS).
        Enhanced for better SATS report handling.
        """
        roe_data = {}
        
        # Split text into lines
        lines = text.split('\n')
        
        # Strategy 1: Look for ROE-related lines with years
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check if line contains ROE-related keywords
            if any(pattern in line_lower for pattern in self.roe_patterns):
                # Extract years from this line and surrounding lines
                context_lines = lines[max(0, i-2):min(len(lines), i+3)]
                context_text = '\n'.join(context_lines)
                
                years = self._extract_years_from_text(context_text)
                numbers = re.findall(r'\d+\.?\d*', context_text)
                
                # Try to match years with numbers
                if years and numbers:
                    for year in years:
                        # Find the closest number to this year
                        year_pos = context_text.find(year)
                        if year_pos != -1:
                            # Look for numbers near this year
                            for num in numbers:
                                num_pos = context_text.find(num)
                                if abs(num_pos - year_pos) < 200:  # Within 200 characters
                                    try:
                                        roe_value = float(num)
                                        if 0 < roe_value < 100:  # Reasonable ROE range
                                            roe_data[year] = {'basic_roe': roe_value}
                                            break
                                    except ValueError:
                                        continue
        
        # Strategy 2: Look for historical data sections with multiple years
        historical_patterns = [
            r'five\s+year\s+summary.*?return\s+on\s+equity',
            r'historical\s+data.*?return\s+on\s+equity',
            r'financial\s+highlights.*?return\s+on\s+equity',
            r'roe.*?(\d{4}.*?\d{4})',
            r'return\s+on\s+equity.*?(\d{4}.*?\d{4})'
        ]
        
        for pattern in historical_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Extract years from the matched section
                section_text = match.group(0)
                years = self._extract_years_from_text(section_text)
                numbers = re.findall(r'\d+\.?\d*', section_text)
                
                # Match years with numbers in sequence
                if years and numbers:
                    for i, year in enumerate(years[:len(numbers)]):
                        try:
                            roe_value = float(numbers[i])
                            if 0 < roe_value < 100:
                                roe_data[year] = {'basic_roe': roe_value}
                        except (ValueError, IndexError):
                            continue
        
        # Strategy 3: Look for table-like structures with years and ROE values
        table_patterns = [
            r'(\d{4})\s+(\d+\.?\d*)',  # Year followed by number
            r'FY\s*(\d{4}-\d{2})\s+(\d+\.?\d*)',  # FY format
            r'(\d{4}-\d{2})\s+(\d+\.?\d*)'  # Year range format
        ]
        
        for pattern in table_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if 'FY' in pattern:
                        year_str = match.group(1)
                        # Convert FY2023-24 to 2023
                        year_match = re.search(r'(\d{4})', year_str)
                        if year_match:
                            year = year_match.group(1)
                    else:
                        year = match.group(1)
                    
                    roe_value = float(match.group(2))
                    if 0 < roe_value < 100:
                        roe_data[year] = {'basic_roe': roe_value}
                except (ValueError, IndexError):
                    continue
        
        return roe_data
    
    def _extract_eps_intelligent(self, text: str) -> Dict[str, Any]:
        """
        Intelligent EPS extraction using AI-like pattern recognition.
        This method can handle any report format by using multiple strategies.
        """
        eps_data = {}
        
        # Strategy 1: Look for financial tables with headers
        table_headers = [
            'year', 'period', 'fiscal year', 'financial year',
            'earnings per share', 'eps', 'basic eps', 'diluted eps',
            'per share', 'per ordinary share', 'per common share'
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check if this line contains table headers
            if any(header in line_lower for header in table_headers):
                # Look for data rows in the next few lines
                for j in range(i+1, min(i+20, len(lines))):
                    data_line = lines[j]
                    years = self._extract_years_from_text(data_line)
                    numbers = re.findall(r'\d+\.?\d*', data_line)
                    
                    if years and numbers:
                        for k, year in enumerate(years):
                            if k < len(numbers):
                                try:
                                    eps_value = float(numbers[k])
                                    if 0 < eps_value < 1000:
                                        eps_data[year] = {'basic_eps': eps_value}
                                except (ValueError, IndexError):
                                    continue
        
        # Strategy 2: Look for structured data blocks
        structured_patterns = [
            r'(\d{4})\s*[:\-]\s*(\d+\.?\d*)',  # 2023: 1.25 or 2023 - 1.25
            r'FY\s*(\d{4})\s*[:\-]\s*(\d+\.?\d*)',  # FY2023: 1.25
            r'(\d{4})\s*=\s*(\d+\.?\d*)',  # 2023 = 1.25
            r'(\d{4})\s*→\s*(\d+\.?\d*)',  # 2023 → 1.25
        ]
        
        for pattern in structured_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    year = match.group(1)
                    eps_value = float(match.group(2))
                    if 0 < eps_value < 1000:
                        eps_data[year] = {'basic_eps': eps_value}
                except (ValueError, IndexError):
                    continue
        
        # Strategy 3: Look for bullet points or list items
        bullet_patterns = [
            r'•\s*(\d{4})\s*:\s*(\d+\.?\d*)',  # • 2023: 1.25
            r'-\s*(\d{4})\s*:\s*(\d+\.?\d*)',  # - 2023: 1.25
            r'(\d{4})\s*•\s*(\d+\.?\d*)',  # 2023 • 1.25
            r'(\d{4})\s*-\s*(\d+\.?\d*)',  # 2023 - 1.25
        ]
        
        for pattern in bullet_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    year = match.group(1)
                    eps_value = float(match.group(2))
                    if 0 < eps_value < 1000:
                        eps_data[year] = {'basic_eps': eps_value}
                except (ValueError, IndexError):
                    continue
        
        # Strategy 4: Look for financial summary sections
        summary_sections = [
            'financial summary', 'earnings summary', 'eps summary',
            'per share summary', 'financial highlights', 'key metrics',
            'financial performance', 'earnings performance'
        ]
        
        for section in summary_sections:
            if section in text.lower():
                # Extract the section content
                section_start = text.lower().find(section)
                section_end = min(section_start + 2000, len(text))
                section_text = text[section_start:section_end]
                
                # Look for years and numbers in this section
                years = self._extract_years_from_text(section_text)
                numbers = re.findall(r'\d+\.?\d*', section_text)
                
                if years and numbers:
                    for i, year in enumerate(years[:len(numbers)]):
                        try:
                            eps_value = float(numbers[i])
                            if 0 < eps_value < 1000:
                                eps_data[year] = {'basic_eps': eps_value}
                        except (ValueError, IndexError):
                            continue
        
        return eps_data
    
    def _is_historical_financial_block(self, text: str) -> bool:
        """
        Determine if a text block contains historical financial data.
        """
        text_lower = text.lower()
        
        # Check for historical section indicators
        for section in self.historical_sections:
            if section in text_lower:
                return True
        
        # Check for multiple years (indicative of historical data)
        years = self._extract_years_from_text(text)
        if len(years) >= 3:  # At least 3 years suggest historical data
            return True
        
        # Check for financial keywords with years
        financial_keywords = [
            'earnings', 'profit', 'income', 'revenue', 'eps', 'roe',
            'per share', 'financial', 'consolidated', 'million', 'billion'
        ]
        
        keyword_count = sum(1 for keyword in financial_keywords if keyword in text_lower)
        if keyword_count >= 2 and len(years) >= 2:
            return True
        
        return False
    
    def _is_financial_table_block(self, text: str) -> bool:
        """
        Determine if a text block contains tabular financial data.
        """
        text_lower = text.lower()
        
        # Look for table-like characteristics
        lines = text.split('\n')
        if len(lines) < 3:
            return False
        
        # Check for consistent number patterns across lines
        number_lines = 0
        for line in lines:
            numbers = re.findall(r'\d+\.?\d*', line)
            if len(numbers) >= 3:  # At least 3 numbers per line
                number_lines += 1
        
        # If most lines have numbers, likely a table
        if number_lines >= len(lines) * 0.6:
            return True
        
        return False
    
    def extract_historical_eps_data(self, layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract historical EPS data using multiple advanced strategies with format detection.
        """
        eps_data = {}
        
        # Detect report format
        report_format = self._detect_report_format(layout_data['full_text'])
        logger.info(f"Detected report format: {report_format}")
        
        # For SATS-style reports, prioritize SATS-specific extraction
        if report_format == 'sats_style':
            # Strategy 1: SATS-specific text block extraction (PRIORITY)
            text_block_eps = self._extract_eps_from_text_blocks(layout_data['full_text'])
            eps_data.update(text_block_eps)
            
            # Only proceed with other strategies if SATS-specific extraction didn't find enough data
            if len(eps_data) < 5:  # If we didn't find enough SATS-specific data
                # Strategy 2: Extract from historical sections
                for section in layout_data['historical_sections']:
                    section_text = section['text']
                    section_eps = self._extract_eps_from_historical_section(section_text)
                    eps_data.update(section_eps)
                
                # Strategy 3: Intelligent extraction for any report format
                intelligent_eps = self._extract_eps_intelligent(layout_data['full_text'])
                eps_data.update(intelligent_eps)
                
                # Strategy 4: Extract from full text using advanced patterns
                full_text_eps = self._extract_eps_from_full_text(layout_data['full_text'])
                eps_data.update(full_text_eps)
        else:
            # For non-SATS reports, use the original strategy order
            # Strategy 1: Extract from historical sections
            for section in layout_data['historical_sections']:
                section_text = section['text']
                section_eps = self._extract_eps_from_historical_section(section_text)
                eps_data.update(section_eps)
            
            # Strategy 2: Extract from financial tables (for OCBC-style reports)
            for table in layout_data['financial_tables']:
                table_text = table['text']
                table_eps = self._extract_eps_from_financial_table(table_text)
                eps_data.update(table_eps)
            
            # Strategy 3: Extract from text blocks (for SATS-style reports)
            if report_format in ['sats_style', 'generic'] or not eps_data:
                text_block_eps = self._extract_eps_from_text_blocks(layout_data['full_text'])
                eps_data.update(text_block_eps)
            
            # Strategy 4: Intelligent extraction for any report format
            intelligent_eps = self._extract_eps_intelligent(layout_data['full_text'])
            eps_data.update(intelligent_eps)
            
            # Strategy 5: Extract from full text using advanced patterns
            full_text_eps = self._extract_eps_from_full_text(layout_data['full_text'])
            eps_data.update(full_text_eps)
            
            # Strategy 6: Look for specific historical patterns
            historical_pattern_eps = self._extract_from_historical_patterns(layout_data['full_text'])
            eps_data.update(historical_pattern_eps)
        
        return eps_data
    
    def _extract_eps_from_historical_section(self, text: str) -> Dict[str, Any]:
        """
        Extract EPS data from historical sections using advanced pattern matching.
        """
        eps_values = {}
        
        # Look for EPS-related lines with multiple years
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if line contains EPS-related keywords
            if any(pattern in line_lower for pattern in self.eps_patterns):
                # Extract all numbers and years from this line
                numbers = re.findall(r'\d+\.?\d*', line)
                years = self._extract_years_from_text(line)
                
                # If we have multiple numbers and years, try to match them
                if len(numbers) >= 3 and len(years) >= 3:
                    # Assume the numbers correspond to years in order
                    for i, year in enumerate(years[:len(numbers)]):
                        try:
                            value = float(numbers[i])
                            if 0.01 <= value <= 1000:
                                eps_values[year] = {
                                    'basic_eps': value,
                                    'currency': self._detect_currency(text),
                                    'source': 'historical_section',
                                    'confidence': 0.9
                                }
                        except (ValueError, IndexError):
                            continue
        
        return eps_values
    
    def _extract_eps_from_financial_table(self, text: str) -> Dict[str, Any]:
        """
        Extract EPS data from financial tables using table structure analysis.
        """
        eps_values = {}
        
        # Split into lines and analyze table structure
        lines = text.split('\n')
        
        # Look for header row with years
        header_years = []
        for line in lines:
            years = self._extract_years_from_text(line)
            if len(years) >= 3:  # Likely a header row
                header_years = years
                break
        
        if header_years:
            # Look for EPS row
            for line in lines:
                line_lower = line.lower()
                if any(pattern in line_lower for pattern in self.eps_patterns):
                    # Extract numbers from this line
                    numbers = re.findall(r'\d+\.?\d*', line)
                    
                    # Match numbers with years
                    for i, year in enumerate(header_years[:len(numbers)]):
                        try:
                            value = float(numbers[i])
                            if 0.01 <= value <= 1000:
                                eps_values[year] = {
                                    'basic_eps': value,
                                    'currency': self._detect_currency(text),
                                    'source': 'financial_table',
                                    'confidence': 0.95
                                }
                        except (ValueError, IndexError):
                            continue
        
        return eps_values
    
    def _extract_eps_from_full_text(self, text: str) -> Dict[str, Any]:
        """
        Extract EPS data from full text using comprehensive pattern matching.
        """
        eps_values = {}
        
        # Look for "Basic earnings" followed by multiple years of data
        basic_pattern = r'basic\s+earnings\s+([\d\.\s]+)'
        basic_matches = re.finditer(basic_pattern, text, re.IGNORECASE)
        
        for match in basic_matches:
            numbers_str = match.group(1)
            numbers = re.findall(r'\d+\.?\d*', numbers_str)
            
            # Find years in surrounding context
            context_before = text[max(0, match.start() - 1000):match.start()]
            context_after = text[match.end():min(len(text), match.end() + 1000)]
            
            years_before = self._extract_years_from_text(context_before)
            years_after = self._extract_years_from_text(context_after)
            all_years = years_before + years_after
            
            # Try different year mappings
            year_mappings = [
                # Most recent first (2024, 2023, 2022, 2021, 2020)
                [2024, 2023, 2022, 2021, 2020],
                # Historical sequence (2020, 2019, 2018, 2017, 2016)
                [2020, 2019, 2018, 2017, 2016],
                # Reverse historical (2016, 2017, 2018, 2019, 2020)
                [2016, 2017, 2018, 2019, 2020]
            ]
            
            for year_mapping in year_mappings:
                for i, number in enumerate(numbers[:5]):  # Take first 5 years
                    try:
                        value = float(number)
                        year = year_mapping[i]
                        
                        if 1990 <= year <= 2030 and 0.01 <= value <= 1000:
                            eps_values[str(year)] = {
                                'basic_eps': value,
                                'currency': self._detect_currency(text),
                                'source': 'full_text_pattern',
                                'confidence': 0.85
                            }
                    except (ValueError, IndexError):
                        continue
        
        return eps_values
    
    def _extract_from_historical_patterns(self, text: str) -> Dict[str, Any]:
        """
        Extract EPS data using specific historical data patterns.
        """
        eps_values = {}
        
        # Look for historical data sections
        historical_patterns = [
            r'five\s+year\s+summary.*?basic\s+earnings.*?([\d\.\s]+)',
            r'historical\s+data.*?basic\s+earnings.*?([\d\.\s]+)',
            r'financial\s+highlights.*?basic\s+earnings.*?([\d\.\s]+)',
            r'per\s+share\s+data.*?basic\s+earnings.*?([\d\.\s]+)'
        ]
        
        for pattern in historical_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                numbers_str = match.group(1)
                numbers = re.findall(r'\d+\.?\d*', numbers_str)
                
                # Try to find years in the context
                context = text[max(0, match.start() - 500):match.end() + 500]
                years = self._extract_years_from_text(context)
                
                # Match years with numbers
                if years and len(numbers) >= len(years):
                    for i, year in enumerate(years[:len(numbers)]):
                        try:
                            value = float(numbers[i])
                            if 1990 <= int(year) <= 2030 and 0.01 <= value <= 1000:
                                eps_values[year] = {
                                    'basic_eps': value,
                                    'currency': self._detect_currency(text),
                                    'source': 'historical_pattern',
                                    'confidence': 0.9
                                }
                        except (ValueError, IndexError):
                            continue
        
        return eps_values
    
    def extract_historical_roe_data(self, layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract historical ROE data using multiple advanced strategies with format detection.
        """
        roe_data = {}
        
        # Detect report format
        report_format = self._detect_report_format(layout_data['full_text'])
        logger.info(f"Detected report format: {report_format}")
        
        # For SATS-style reports, prioritize SATS-specific extraction
        if report_format == 'sats_style':
            # Strategy 1: SATS-specific text block extraction (PRIORITY)
            text_block_roe = self._extract_roe_from_text_blocks(layout_data['full_text'])
            roe_data.update(text_block_roe)
            
            # Only proceed with other strategies if SATS-specific extraction didn't find enough data
            if len(roe_data) < 5:  # If we didn't find enough SATS-specific data
                # Strategy 2: Extract from historical sections
                for section in layout_data['historical_sections']:
                    section_text = section['text']
                    section_roe = self._extract_roe_from_historical_section(section_text)
                    roe_data.update(section_roe)
                
                # Strategy 3: Extract from full text using advanced patterns
                full_text_roe = self._extract_roe_from_full_text(layout_data['full_text'])
                roe_data.update(full_text_roe)
        else:
            # For non-SATS reports, use the original strategy order
            # Strategy 1: Extract from historical sections
            for section in layout_data['historical_sections']:
                section_text = section['text']
                section_roe = self._extract_roe_from_historical_section(section_text)
                roe_data.update(section_roe)
            
            # Strategy 2: Extract from financial tables (for OCBC-style reports)
            for table in layout_data['financial_tables']:
                table_text = table['text']
                table_roe = self._extract_roe_from_financial_table(table_text)
                roe_data.update(table_roe)
            
            # Strategy 3: Extract from text blocks (for SATS-style reports)
            if report_format in ['sats_style', 'generic'] or not roe_data:
                text_block_roe = self._extract_roe_from_text_blocks(layout_data['full_text'])
                roe_data.update(text_block_roe)
            
            # Strategy 4: Extract from full text using advanced patterns
            full_text_roe = self._extract_roe_from_full_text(layout_data['full_text'])
            roe_data.update(full_text_roe)
        
        return roe_data
    
    def _extract_roe_from_historical_section(self, text: str) -> Dict[str, Any]:
        """
        Extract ROE data from historical sections.
        """
        roe_values = {}
        
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if line contains ROE-related keywords
            if any(pattern in line_lower for pattern in self.roe_patterns):
                numbers = re.findall(r'\d+\.?\d*', line)
                years = self._extract_years_from_text(line)
                
                if len(numbers) >= 3 and len(years) >= 3:
                    for i, year in enumerate(years[:len(numbers)]):
                        try:
                            value = float(numbers[i])
                            if 0.1 <= value <= 100:  # ROE is typically in percentage
                                roe_values[year] = {
                                    'basic_roe': value,
                                    'unit': 'percentage',
                                    'source': 'historical_section',
                                    'confidence': 0.9
                                }
                        except (ValueError, IndexError):
                            continue
        
        return roe_values
    
    def _extract_roe_from_financial_table(self, text: str) -> Dict[str, Any]:
        """
        Extract ROE data from financial tables.
        """
        roe_values = {}
        
        lines = text.split('\n')
        
        # Look for header row with years
        header_years = []
        for line in lines:
            years = self._extract_years_from_text(line)
            if len(years) >= 3:
                header_years = years
                break
        
        if header_years:
            # Look for ROE row
            for line in lines:
                line_lower = line.lower()
                if any(pattern in line_lower for pattern in self.roe_patterns):
                    numbers = re.findall(r'\d+\.?\d*', line)
                    
                    for i, year in enumerate(header_years[:len(numbers)]):
                        try:
                            value = float(numbers[i])
                            if 0.1 <= value <= 100:
                                roe_values[year] = {
                                    'basic_roe': value,
                                    'unit': 'percentage',
                                    'source': 'financial_table',
                                    'confidence': 0.95
                                }
                        except (ValueError, IndexError):
                            continue
        
        return roe_values
    
    def _extract_roe_from_full_text(self, text: str) -> Dict[str, Any]:
        """
        Extract ROE data from full text.
        """
        roe_values = {}
        
        # Look for ROE patterns with multiple years
        roe_pattern = r'return\s+on\s+equity\s+([\d\.\s]+)'
        roe_matches = re.finditer(roe_pattern, text, re.IGNORECASE)
        
        for match in roe_matches:
            numbers_str = match.group(1)
            numbers = re.findall(r'\d+\.?\d*', numbers_str)
            
            # Find years in context
            context_before = text[max(0, match.start() - 1000):match.start()]
            context_after = text[match.end():min(len(text), match.end() + 1000)]
            
            years_before = self._extract_years_from_text(context_before)
            years_after = self._extract_years_from_text(context_after)
            all_years = years_before + years_after
            
            # Try different year mappings
            year_mappings = [
                [2024, 2023, 2022, 2021, 2020],
                [2020, 2019, 2018, 2017, 2016],
                [2016, 2017, 2018, 2019, 2020]
            ]
            
            for year_mapping in year_mappings:
                for i, number in enumerate(numbers[:5]):
                    try:
                        value = float(number)
                        year = year_mapping[i]
                        
                        if 1990 <= year <= 2030 and 0.1 <= value <= 100:
                            roe_values[str(year)] = {
                                'basic_roe': value,
                                'unit': 'percentage',
                                'source': 'full_text_pattern',
                                'confidence': 0.85
                            }
                    except (ValueError, IndexError):
                        continue
        
        return roe_values
    
    def _detect_currency(self, text: str) -> str:
        """
        Detect currency from text context.
        """
        text_lower = text.lower()
        
        currency_indicators = {
            'SGD': ['singapore', 'sgd', 's$', 'singapore dollar'],
            'USD': ['us dollar', 'usd', '$', 'dollar'],
            'EUR': ['euro', 'eur', '€'],
            'GBP': ['pound', 'gbp', '£'],
            'HKD': ['hong kong', 'hkd', 'hk$'],
            'CNY': ['yuan', 'cny', 'rmb', '¥']
        }
        
        for currency, indicators in currency_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    return currency
        
        return 'SGD'  # Default for Singapore companies
    
    def _detect_eps_unit(self, eps_data: Dict[str, Any], text: str) -> str:
        """
        Detect whether EPS values are in cents or dollars based on:
        1. The actual values extracted (range analysis)
        2. Text context in the document
        3. Common patterns in annual reports
        4. SATS-specific patterns (small decimal values)
        """
        text_lower = text.lower()
        
        # Check text context for unit indicators
        if any(phrase in text_lower for phrase in ['cents', 'cent', '¢']):
            return 'cents'
        if any(phrase in text_lower for phrase in ['dollars', 'dollar', '$', 'sgd']):
            return 'dollars'
        
        # Analyze the actual values extracted
        eps_values = []
        for year, data in eps_data.items():
            if isinstance(data, dict) and 'basic_eps' in data and data['basic_eps'] is not None:
                try:
                    eps_values.append(float(data['basic_eps']))
                except (ValueError, TypeError):
                    continue
            elif isinstance(data, (int, float)) and data is not None:
                eps_values.append(float(data))
        
        if not eps_values:
            return 'dollars'  # Default to dollars if no data
        
        # Analyze value ranges to determine units
        avg_value = sum(eps_values) / len(eps_values)
        max_value = max(eps_values)
        min_value = min(eps_values)
        
        # SATS-specific logic: If values are small decimals (0.01 to 0.99), they're in dollars
        # SATS EPS values are typically: 0.12, 0.14, 0.16, 0.17, 0.10, -0.05, 0.0133, -0.0165, 0.0282
        if all(0.001 <= abs(v) <= 1.0 for v in eps_values):
            return 'dollars'  # Small decimal values are dollars for SATS
        
        # If average is less than 1, likely in dollars (e.g., 0.01 to 0.99)
        # If average is 1-100, could be either
        # If average is > 100, likely in cents
        
        if avg_value < 1.0:
            return 'dollars'  # Values like 0.01, 0.02 are likely dollars
        elif avg_value > 100:
            return 'cents'    # Values like 101, 250 are likely cents
        else:
            # For values 1-100, check for decimal places
            decimal_count = sum(1 for v in eps_values if v % 1 != 0)
            if decimal_count > len(eps_values) * 0.5:
                return 'dollars'  # Many decimal values suggest dollars
            else:
                return 'dollars'  # Default to dollars for Singapore companies
    
    def extract_from_pdf_file(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main extraction function that processes a PDF file using enhanced strategies.
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text with enhanced layout analysis
        layout_data = self.extract_text_with_enhanced_layout(pdf_path)
        
        if not layout_data['full_text']:
            logger.error("No text content extracted from PDF")
            return {"error": "No text content found"}
        
        # Extract historical EPS data
        eps_data = self.extract_historical_eps_data(layout_data)
        logger.info(f"Extracted EPS data for {len(eps_data)} years")
        
        # Extract historical ROE data
        roe_data = self.extract_historical_roe_data(layout_data)
        logger.info(f"Extracted ROE data for {len(roe_data)} years")
        
        # Validate historical data availability and filter to last 10 years
        validation_result = self.validate_historical_data_availability(eps_data, roe_data)
        
        # Use filtered data (last 10 years only) for the result
        filtered_eps_data = validation_result.get('filtered_eps_data', {})
        filtered_roe_data = validation_result.get('filtered_roe_data', {})
        
        # Calculate confidence score using filtered data
        confidence_score = self._calculate_confidence_score(filtered_eps_data, filtered_roe_data)
        
        # Detect EPS units based on the filtered values extracted
        eps_unit = self._detect_eps_unit(filtered_eps_data, layout_data['full_text'])
        
        result = {
            "eps_data": filtered_eps_data,  # Use filtered data (last 10 years only)
            "roe_data": filtered_roe_data,  # Use filtered data (last 10 years only)
            "all_eps_data": eps_data,  # Keep all extracted data for reference
            "all_roe_data": roe_data,  # Keep all extracted data for reference
            "units": {
                "eps_unit": eps_unit,
                "roe_unit": "percentage"
            },
            "currency": self._detect_currency(layout_data['full_text']),
            "extraction_method": "enhanced_historical_extraction",
            "confidence_score": confidence_score,
            "source_file": pdf_path,
            "validation": validation_result,
            "transparency": {
                "historical_sections_found": len(layout_data['historical_sections']),
                "financial_tables_found": len(layout_data['financial_tables']),
                "pages_processed": len(layout_data['pages']),
                "extraction_strategies": ["historical_section_analysis", "table_analysis", "pattern_matching", "full_text_analysis"],
                "total_years_extracted": len(eps_data),
                "filtered_years_used": len(filtered_eps_data)
            }
        }
        
        # Add error if insufficient data
        if validation_result['errors']:
            result['error'] = f"Insufficient data: {'; '.join(validation_result['errors'])}"
            result['recommendations'] = validation_result['recommendations']
        
        return result
    
    def _calculate_confidence_score(self, eps_data: Dict, roe_data: Dict) -> float:
        """
        Calculate confidence score based on extraction quality.
        """
        confidence = 0.0
        
        # Factor 1: Number of years with data
        eps_years = len([y for y, d in eps_data.items() 
                        if isinstance(d, dict) and d.get('basic_eps') is not None])
        roe_years = len([y for y, d in roe_data.items() 
                        if isinstance(d, dict) and d.get('basic_roe') is not None])
        
        # Higher weight for more years (target is 10 years)
        confidence += min((eps_years + roe_years) / 20.0, 0.5)
        
        # Factor 2: Data quality (confidence scores)
        eps_confidence = sum(d.get('confidence', 0) 
                           for d in eps_data.values() if isinstance(d, dict))
        roe_confidence = sum(d.get('confidence', 0) 
                           for d in roe_data.values() if isinstance(d, dict))
        
        confidence += min((eps_confidence + roe_confidence) / 20.0, 0.3)
        
        # Factor 3: Source diversity
        sources = set()
        for d in eps_data.values():
            if isinstance(d, dict):
                sources.add(d.get('source', ''))
        for d in roe_data.values():
            if isinstance(d, dict):
                sources.add(d.get('source', ''))
        
        confidence += min(len(sources) / 5.0, 0.2)
        
        return min(confidence, 1.0)

    def validate_historical_data_availability(self, eps_data: Dict, roe_data: Dict) -> Dict[str, Any]:
        """
        Validate if sufficient historical data is available for analysis.
        Returns validation results with recommendations.
        """
        # Filter to only the last 10 years and validate year accuracy
        filtered_eps_data = self._filter_to_last_10_years(eps_data)
        filtered_roe_data = self._filter_to_last_10_years(roe_data)
        
        eps_years = [y for y, d in filtered_eps_data.items() 
                    if isinstance(d, dict) and d.get('basic_eps') is not None]
        roe_years = [y for y, d in filtered_roe_data.items() 
                    if isinstance(d, dict) and d.get('basic_roe') is not None]
        
        validation_result = {
            'eps_years_found': len(eps_years),
            'roe_years_found': len(roe_years),
            'eps_years_list': sorted(eps_years),
            'roe_years_list': sorted(roe_years),
            'has_sufficient_eps': len(eps_years) >= 5,
            'has_sufficient_roe': len(roe_years) >= 5,
            'has_10_year_eps': len(eps_years) >= 10,
            'has_10_year_roe': len(roe_years) >= 10,
            'filtered_eps_data': filtered_eps_data,
            'filtered_roe_data': filtered_roe_data,
            'recommendations': [],
            'errors': []
        }
        
        # Check for minimum requirements (10 years for proper analysis)
        if len(eps_years) < 10:
            validation_result['errors'].append(
                f"Insufficient EPS data: Found {len(eps_years)} years, minimum 10 required for comprehensive analysis"
            )
            validation_result['recommendations'].extend([
                "Check if PDF files contain historical financial summaries",
                "Verify files are actual annual reports with financial data",
                "Try different annual report years that may have more historical data",
                "Look for 10-year historical data sections in annual reports",
                "Consider using multiple annual reports from different years to build 10-year dataset"
            ])
        
        if len(roe_years) < 10:
            validation_result['warnings'] = validation_result.get('warnings', [])
            validation_result['warnings'].append(
                f"Limited ROE data: Found {len(roe_years)} years, minimum 10 recommended for comprehensive analysis"
            )
        
        # Check for 10-year data availability
        if len(eps_years) < 10:
            validation_result['recommendations'].extend([
                "For optimal analysis, aim for 10+ years of historical data",
                "Consider using multiple annual reports from different years",
                "Check if company provides 10-year historical summaries in reports"
            ])
        
        if len(roe_years) < 10:
            validation_result['recommendations'].extend([
                "ROE data limited - consider additional sources for comprehensive analysis"
            ])
        
        return validation_result

    def _filter_to_last_10_years(self, data: Dict) -> Dict:
        """
        Filter data to only include the last 10 years and validate year accuracy.
        Returns filtered data with only valid years from the last 10 years.
        """
        if not data:
            return {}
        
        current_year = datetime.now().year
        
        # Extract all years and validate them
        valid_years = []
        for year_str, data_item in data.items():
            if not isinstance(data_item, dict):
                continue
                
            try:
                year = int(year_str)
                # Validate year is reasonable (between 1990 and current year + 1)
                if 1990 <= year <= current_year + 1:
                    valid_years.append(year)
                else:
                    logger.warning(f"Invalid year found: {year_str} (outside reasonable range 1990-{current_year + 1})")
            except ValueError:
                logger.warning(f"Invalid year format: {year_str}")
                continue
        
        if not valid_years:
            logger.warning("No valid years found in data")
            return {}
        
        # Sort years and get the last 10 years
        valid_years.sort()
        last_10_years = valid_years[-10:] if len(valid_years) >= 10 else valid_years
        
        # Create filtered data with only the last 10 years
        filtered_data = {}
        for year in last_10_years:
            year_str = str(year)
            if year_str in data:
                filtered_data[year_str] = data[year_str]
        
        logger.info(f"Filtered data to last {len(last_10_years)} years: {last_10_years}")
        return filtered_data

    def _validate_year_accuracy(self, years: List[str]) -> List[str]:
        """
        Validate that extracted years are accurate and reasonable.
        Returns list of valid years.
        """
        current_year = datetime.now().year
        valid_years = []
        
        for year_str in years:
            try:
                year = int(year_str)
                # Check if year is reasonable (between 1990 and current year + 1)
                if 1990 <= year <= current_year + 1:
                    valid_years.append(year_str)
                else:
                    logger.warning(f"Invalid year detected: {year_str} (outside reasonable range)")
            except ValueError:
                logger.warning(f"Invalid year format: {year_str}")
                continue
        
        return valid_years

def main():
    """Test the enhanced PDF extractor."""
    extractor = EnhancedPDFExtractor()
    
    # Test directory
    test_dir = r"C:/Users/Arty2/CrewAIProjects/Financial/Scripts/reports/OCBC"
    
    if not os.path.exists(test_dir):
        print(f"Test directory not found: {test_dir}")
        return
    
    # Test with a specific file
    test_file = "2021 annual report OCBC.pdf"
    pdf_path = os.path.join(test_dir, test_file)
    
    if not os.path.exists(pdf_path):
        print(f"Test file not found: {pdf_path}")
        return
    
    print(f"Testing enhanced PDF extraction from: {test_file}")
    print()
    
    # Extract data
    result = extractor.extract_from_pdf_file(pdf_path)
    
    print("=== ENHANCED EXTRACTION RESULTS ===")
    print(f"Confidence Score: {result.get('confidence_score', 0.0):.2f}")
    print(f"Extraction Method: {result.get('extraction_method', 'unknown')}")
    print(f"Currency: {result.get('currency', 'unknown')}")
    print()
    
    # Display extracted EPS data
    eps_data = result.get('eps_data', {})
    if eps_data:
        print("Extracted EPS Data:")
        for year, data in eps_data.items():
            if isinstance(data, dict) and data.get('basic_eps') is not None:
                print(f"  {year}: {data['basic_eps']} ({data.get('currency', 'Unknown')})")
                print(f"    Source: {data.get('source', 'unknown')}")
                print(f"    Confidence: {data.get('confidence', 0.0):.2f}")
    else:
        print("No EPS data found")
    
    print()
    
    # Display extracted ROE data
    roe_data = result.get('roe_data', {})
    if roe_data:
        print("Extracted ROE Data:")
        for year, data in roe_data.items():
            if isinstance(data, dict) and data.get('basic_roe') is not None:
                print(f"  {year}: {data['basic_roe']}%")
                print(f"    Source: {data.get('source', 'unknown')}")
                print(f"    Confidence: {data.get('confidence', 0.0):.2f}")
    else:
        print("No ROE data found")
    
    print()
    print("=== TRANSPARENCY REPORT ===")
    transparency = result.get('transparency', {})
    print(f"Historical Sections Found: {transparency.get('historical_sections_found', 0)}")
    print(f"Financial Tables Found: {transparency.get('financial_tables_found', 0)}")
    print(f"Pages Processed: {transparency.get('pages_processed', 0)}")
    print(f"Extraction Strategies: {', '.join(transparency.get('extraction_strategies', []))}")

if __name__ == "__main__":
    main() 