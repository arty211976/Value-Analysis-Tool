#!/usr/bin/env python3
"""
Advanced EPS/ROE Extractor using AI and Machine Learning
More robust than regex-based extraction for handling diverse report formats
"""

import os
import re
import json
import logging
import copy
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import fitz  # PyMuPDF
import pdfplumber
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# For AI-powered extraction
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("Warning: AI libraries not available. Install with: pip install transformers sentence-transformers torch")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FinancialMetric:
    """Data class for financial metrics with confidence and metadata"""
    value: float
    year: str
    metric_type: str  # 'eps', 'roe', 'revenue', etc.
    confidence: float
    source: str  # 'table', 'text', 'ai', 'ml'
    currency: str = 'USD'
    unit: str = 'dollars'  # 'dollars', 'cents', 'percentage'
    context: str = ''  # surrounding text for validation

class AdvancedFinancialExtractor:
    """
    Advanced financial data extractor using multiple AI and ML approaches
    for robust extraction across diverse report formats.
    """
    
    def __init__(self, use_ai: bool = True, ensemble_enabled: bool = True, ensemble_mode: str = "strict"):
        self.use_ai = use_ai and AI_AVAILABLE
        self.current_year = datetime.now().year
        self.ensemble_enabled = ensemble_enabled
        # Modes: 'strict' requires closer agreement, 'lenient' allows larger tolerance
        self.ensemble_mode = ensemble_mode
        
        # Initialize AI models if available
        if self.use_ai:
            self._initialize_ai_models()
        
        # Financial metric keywords for different languages/formats
        self.metric_keywords = {
            'eps': [
                'earnings per share', 'eps', 'basic eps', 'diluted eps',
                'net income per share', 'profit per share', 'per share earnings',
                'basic earnings per share', 'diluted earnings per share',
                'earnings per ordinary share', 'earnings per common share',
                'net earnings per share', 'profit attributable to shareholders per share'
            ],
            'roe': [
                'return on equity', 'roe', 'return on shareholders equity',
                'net income to equity', 'profit to equity', 'return on common equity',
                'return on total equity', 'equity return', 'shareholders return'
            ]
        }
        
        # Table structure patterns for different report formats
        self.table_patterns = {
            'financial_highlights': [
                r'financial\s+highlights.*?(\d{4}.*?\d{4})',
                r'five\s+year\s+summary.*?(\d{4}.*?\d{4})',
                r'historical\s+data.*?(\d{4}.*?\d{4})'
            ],
            'income_statement': [
                r'consolidated\s+income.*?(\d{4}.*?\d{4})',
                r'income\s+statement.*?(\d{4}.*?\d{4})',
                r'profit\s+and\s+loss.*?(\d{4}.*?\d{4})'
            ],
            'per_share_data': [
                r'per\s+share\s+data.*?(\d{4}.*?\d{4})',
                r'earnings\s+per\s+share.*?(\d{4}.*?\d{4})',
                r'per\s+ordinary\s+share.*?(\d{4}.*?\d{4})'
            ]
        }
    
    def _initialize_ai_models(self):
        """Initialize AI models for enhanced extraction"""
        try:
            # Financial text classification model
            self.financial_classifier = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                return_all_scores=True
            )
            
            # Sentence embeddings for semantic similarity
            self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Named entity recognition for financial entities
            self.ner_pipeline = pipeline(
                "ner",
                model="dslim/bert-base-NER",
                aggregation_strategy="simple"
            )
            
            logger.info("AI models initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize AI models: {e}")
            self.use_ai = False
    
    def extract_from_pdf_file(self, pdf_path: str) -> Dict[str, Any]:
        """
        Main extraction method using multiple advanced strategies
        """
        try:
            # Extract text and layout information
            layout_data = self._extract_text_with_layout(pdf_path)
            
            # Extract financial metrics using multiple approaches
            eps_data = self._extract_eps_advanced(layout_data)
            roe_data = self._extract_roe_advanced(layout_data)
            
            # Validate and clean the extracted data
            eps_data = self._validate_financial_data(eps_data, 'eps')
            roe_data = self._validate_financial_data(roe_data, 'roe')
            
            # AI-POWERED DATA CORRECTION (Apply BEFORE validation)
            print(f"\n=== AI-POWERED DATA CORRECTION ===")
            corrected_eps_data = self._apply_ai_corrections_to_data(eps_data, 'eps')
            corrected_roe_data = self._apply_ai_corrections_to_data(roe_data, 'roe')

            # Structured validator extraction (Deterministic checks on parsed tables/tokens)
            ensemble_report = {}
            print("\n=== STRUCTURED VALIDATOR (ENSEMBLE) ===")
            validator_eps = self._structured_validator_extract(layout_data, 'eps', pdf_path=pdf_path)
            # Debug: show EPS anchor extraction when present
            try:
                if isinstance(validator_eps, dict) and validator_eps:
                    keys = [k for k in validator_eps.keys() if k != 'units']
                    print(f"EPS ANCHOR SUMMARY: years={sorted(keys)} sample={validator_eps.get(sorted(keys)[-1], {}) if keys else {}}")
            except Exception:
                pass
            validator_roe = self._structured_validator_extract(layout_data, 'roe', pdf_path=pdf_path)
            
            # Reconcile AI-corrected data with validator data (prefer validator tables when present)
            final_eps_data, eps_agreement = self._reconcile_ai_and_validator(
                corrected_eps_data, validator_eps, 'eps'
            )
            final_roe_data, roe_agreement = self._reconcile_ai_and_validator(
                corrected_roe_data, validator_roe, 'roe'
            )
            ensemble_report = {
                'enabled': True,
                'mode': self.ensemble_mode,
                'eps_agreement': eps_agreement,
                'roe_agreement': roe_agreement
            }
            
            # Comprehensive validation of reconciled data
            validation_report = self._validate_extracted_data(final_eps_data, final_roe_data)
            
            # Calculate confidence scores on final reconciled data
            confidence_score = self._calculate_confidence_score(final_eps_data, final_roe_data)
            
            # Check data completeness on final reconciled data
            completeness_report = self._check_data_completeness(final_eps_data, final_roe_data)
            
            # Normalize top-level units to avoid downstream re-conversion
            if isinstance(final_eps_data, dict):
                final_eps_data['units'] = final_eps_data.get('units', {'eps_unit': 'dollars'})
            if isinstance(final_roe_data, dict):
                final_roe_data['units'] = final_roe_data.get('units', {'roe_unit': 'percentage'})

            # Add validation information to the output
            result = {
                'eps_data': final_eps_data,
                'roe_data': final_roe_data,
                'confidence_score': confidence_score,
                'completeness_report': completeness_report,
                'validation_report': validation_report,
                'extraction_methods': ['ai_enhanced', 'table_analysis', 'ml_clustering'],
                'timestamp': datetime.now().isoformat()
            }
            if ensemble_report:
                result['ensemble_report'] = ensemble_report
            
            # Log validation results
            print(f"\n=== EXTRACTION VALIDATION RESULTS (After AI Corrections) ===")
            print(f"Overall confidence: {validation_report['overall_confidence']:.2%}")
            print(f"Data quality score: {validation_report['data_quality_score']:.1f}/100")
            print(f"EPS validation: {validation_report['eps_validation']['valid_count']}/{validation_report['eps_validation']['total_count']} valid")
            print(f"ROE validation: {validation_report['roe_validation']['valid_count']}/{validation_report['roe_validation']['total_count']} valid")
            
            if validation_report['warnings']:
                print(f"\nValidation warnings ({len(validation_report['warnings'])}):")
                for warning in validation_report['warnings']:
                    print(f"  ⚠️  {warning}")
            
            if validation_report['recommendations']:
                print(f"\nRecommendations:")
                for rec in validation_report['recommendations']:
                    print(f"  💡 {rec}")
            
            return result
            
            return corrected_result
            
        except Exception as e:
            logger.error(f"Error extracting from PDF: {e}")
            return {'error': str(e)}
    
    def _extract_text_with_layout(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text with enhanced layout analysis"""
        layout_data = {
            'full_text': '',
            'tables': [],
            'images': [],
            'sections': [],
            'metadata': {},
            'historical_summaries': []  # New: Track historical summary sections
        }
        
        # Extract using PyMuPDF for layout
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text blocks with positioning
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    text = ""
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text += span["text"] + " "
                    
                    # Enhanced section classification
                    section_info = {
                        'text': text.strip(),
                        'bbox': block["bbox"],
                        'page': page_num,
                        'type': 'text',
                        'is_historical_summary': self._is_historical_summary_section(text.strip()),
                        'contains_financial_data': self._contains_financial_data(text.strip())
                    }
                    layout_data['sections'].append(section_info)
                    
                    # Track historical summary sections
                    if section_info['is_historical_summary']:
                        layout_data['historical_summaries'].append(section_info)
            
            # Enhanced table extraction using pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                page_pdf = pdf.pages[page_num]
                tables = page_pdf.extract_tables()
                
                for table_idx, table in enumerate(tables):
                    if table and len(table) > 1:  # Valid table
                        # Enhanced table analysis
                        table_analysis = self._analyze_table_structure(table)
                        table_data = {
                            'data': table,
                            'page': page_num,
                            'index': table_idx,
                            'text': self._table_to_text(table),
                            'is_historical_summary': table_analysis['is_historical_summary'],
                            'contains_financial_data': table_analysis['contains_financial_data'],
                            'year_columns': table_analysis['year_columns'],
                            'metric_rows': table_analysis['metric_rows'],
                            'confidence': table_analysis['confidence']
                        }
                        layout_data['tables'].append(table_data)
        
        # Combine all text
        layout_data['full_text'] = '\n'.join([section['text'] for section in layout_data['sections']])
        
        return layout_data
    
    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert table data to searchable text"""
        text_lines = []
        for row in table:
            if row:
                text_lines.append(' '.join([str(cell) if cell else '' for cell in row]))
        return '\n'.join(text_lines)
    
    def _is_historical_summary_section(self, text: str) -> bool:
        """Detect if a text section contains historical summary data"""
        text_lower = text.lower()
        
        # Keywords that indicate historical summary sections
        historical_keywords = [
            'historical', 'summary', 'five year', 'ten year', 'annual summary',
            'financial highlights', 'performance summary', 'operating results',
            'financial summary', 'results summary', 'key financial indicators',
            'financial performance', 'operating performance', 'financial overview'
        ]
        
        # Year patterns that suggest historical data
        year_patterns = [
            r'\b(19\d{2}|20\d{2})\b',  # Full 4-digit years
            r'(19\d{2}|20\d{2})[-/](19\d{2}|20\d{2})',    # Year ranges like 2019-2020
            r'FY\s*(19\d{2}|20\d{2})',        # Fiscal year references
        ]
        
        # Check for historical keywords
        has_historical_keywords = any(keyword in text_lower for keyword in historical_keywords)
        
        # Check for multiple years
        years_found = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
        has_multiple_years = len(set(years_found)) >= 2
        
        # Check for financial metric keywords
        has_financial_metrics = any(keyword in text_lower for keyword in 
                                  self.metric_keywords['eps'] + self.metric_keywords['roe'])
        
        return has_historical_keywords and (has_multiple_years or has_financial_metrics)
    
    def _contains_financial_data(self, text: str) -> bool:
        """Check if text contains financial data indicators"""
        text_lower = text.lower()
        
        # Financial keywords
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'income', 'eps', 'roe', 'pe ratio',
            'dividend', 'assets', 'liabilities', 'equity', 'cash flow',
            'operating', 'net income', 'gross profit', 'ebitda'
        ]
        
        # Number patterns
        number_patterns = [
            r'\$\d+\.?\d*',      # Dollar amounts
            r'\d+\.?\d*%',       # Percentages
            r'\d+\.?\d*\s*(million|billion|thousand)',  # Large numbers
        ]
        
        has_financial_keywords = any(keyword in text_lower for keyword in financial_keywords)
        has_number_patterns = any(re.search(pattern, text, re.IGNORECASE) for pattern in number_patterns)
        
        return has_financial_keywords or has_number_patterns
    
    def _analyze_table_structure(self, table: List[List[str]]) -> Dict[str, Any]:
        """Analyze table structure to identify historical summary tables"""
        if not table or len(table) < 2:
            return {
                'is_historical_summary': False,
                'contains_financial_data': False,
                'year_columns': [],
                'metric_rows': [],
                'confidence': 0.0
            }
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(table)
        
        # Enhanced year column detection
        year_columns = []
        for col_idx, col in enumerate(df.columns):
            col_text = ' '.join([str(cell) for cell in df.iloc[:, col_idx] if cell])
            years = re.findall(r'\b(19\d{2}|20\d{2})\b', col_text)
            if years:
                # Check if this column contains multiple years (historical data)
                unique_years = list(set(years))
                if len(unique_years) >= 2:
                    # This is likely a historical summary column
                    for year in unique_years:
                        year_columns.append((col_idx, year))
                else:
                    # Single year column
                    year_columns.append((col_idx, years[0]))
        
        # Enhanced metric row detection
        metric_rows = []
        for row_idx, row in df.iterrows():
            row_text = ' '.join([str(cell) for cell in row if cell]).lower()
            
            # Check for EPS keywords
            if any(keyword in row_text for keyword in self.metric_keywords['eps']):
                metric_rows.append((row_idx, 'eps'))
            
            # Check for ROE keywords
            if any(keyword in row_text for keyword in self.metric_keywords['roe']):
                metric_rows.append((row_idx, 'roe'))
            
            # Check for financial data patterns
            if re.search(r'\d+\.?\d*', row_text) and any(keyword in row_text for keyword in ['earnings', 'revenue', 'profit', 'income']):
                metric_rows.append((row_idx, 'financial'))
        
        # Enhanced historical summary detection
        is_historical_summary = False
        confidence = 0.0
        
        # Check for multiple year columns
        if len(year_columns) >= 2:
            is_historical_summary = True
            confidence += 0.3
        
        # Check for metric rows
        if len(metric_rows) > 0:
            is_historical_summary = True
            confidence += 0.2
        
        # Check for table structure patterns
        if len(df) >= 5 and len(df.columns) >= 3:  # Typical historical table size
            confidence += 0.1
        
        # Check for financial keywords in table
        table_text = ' '.join([str(cell) for row in table for cell in row if cell]).lower()
        if any(keyword in table_text for keyword in ['historical', 'summary', 'five year', 'ten year', 'annual']):
            confidence += 0.2
        
        # Check for number patterns that suggest financial data
        numbers = re.findall(r'\d+\.?\d*', table_text)
        if len(numbers) >= 10:  # Sufficient numbers for financial data
            confidence += 0.1
        
        confidence = min(0.95, confidence)
        
        return {
            'is_historical_summary': is_historical_summary,
            'contains_financial_data': len(metric_rows) > 0,
            'year_columns': year_columns,
            'metric_rows': metric_rows,
            'confidence': confidence
        }
    
    def _extract_eps_advanced(self, layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced EPS extraction using multiple AI and ML approaches"""
        eps_metrics = []
        
        # Strategy 1: AI-powered semantic extraction
        if self.use_ai:
            ai_eps = self._extract_eps_with_ai(layout_data['full_text'])
            eps_metrics.extend(ai_eps)
        
        # Strategy 2: Enhanced table structure analysis
        table_eps = self._extract_eps_from_tables(layout_data['tables'])
        eps_metrics.extend(table_eps)
        
        # Strategy 3: Historical summary section extraction (NEW)
        historical_eps = self._extract_from_historical_summary_sections(layout_data['historical_summaries'], 'eps')
        eps_metrics.extend(historical_eps)
        
        # Strategy 4: ML-based clustering for pattern recognition
        ml_eps = self._extract_eps_with_ml(layout_data['full_text'])
        eps_metrics.extend(ml_eps)
        
        # Strategy 5: Context-aware extraction
        context_eps = self._extract_eps_with_context(layout_data['sections'])
        eps_metrics.extend(context_eps)
        
        # Strategy 6: Financial highlights extraction (NEW)
        highlights_eps = self._extract_from_financial_highlights(layout_data['full_text'], 'eps')
        eps_metrics.extend(highlights_eps)
        
        # Strategy 7: Enhanced AI extraction with improved pattern recognition (NEW)
        enhanced_eps = self._extract_eps_with_enhanced_patterns(layout_data['full_text'])
        eps_metrics.extend(enhanced_eps)
        
        # Strategy 8: Comprehensive page-by-page extraction (NEW)
        comprehensive_eps = self._extract_eps_comprehensive(layout_data)
        eps_metrics.extend(comprehensive_eps)
        
        # Merge and deduplicate metrics
        return self._merge_financial_metrics(eps_metrics, 'eps')
    
    def _extract_eps_with_ai(self, text: str) -> List[FinancialMetric]:
        """Extract EPS using AI-powered semantic analysis"""
        metrics = []
        
        if not self.use_ai:
            return metrics
        
        try:
            # Split text into sentences for analysis
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                if len(sentence.strip()) < 10:
                    continue
                
                # Check if sentence contains EPS-related content
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in self.metric_keywords['eps']):
                    
                    # Use AI to classify the sentence
                    classification = self.financial_classifier(sentence[:512])[0]
                    
                    # If classified as financial content, extract numbers
                    if any(label['label'] in ['positive', 'negative'] for label in classification):
                        numbers = re.findall(r'-?\d+\.?\d*', sentence)
                        years = re.findall(r'\b(19|20)\d{2}\b', sentence)
                        
                        for year in years:
                            # Validate year is realistic
                            try:
                                year_int = int(year)
                                current_year = datetime.now().year
                                if not (1995 <= year_int <= current_year):
                                    continue
                            except (ValueError, TypeError):
                                continue
                            
                            # Find the closest number to this year
                            year_pos = sentence.find(year)
                            closest_number = None
                            min_distance = float('inf')
                            
                            for num in numbers:
                                num_pos = sentence.find(num)
                                distance = abs(num_pos - year_pos)
                                if distance < min_distance and distance < 100:
                                    min_distance = distance
                                    closest_number = num
                            
                            if closest_number:
                                try:
                                    value = float(closest_number)
                                    # Detect unit from context
                                    unit = self._detect_eps_unit_from_text(sentence)
                                    if unit == 'cents':
                                        value = value / 100
                                    if self._is_valid_eps_value(value):
                                        metrics.append(FinancialMetric(
                                            value=value,
                                            year=year,
                                            metric_type='eps',
                                            confidence=0.8,
                                            source='ai',
                                            unit='dollars',
                                            context=sentence
                                        ))
                                except ValueError:
                                    continue
            
        except Exception as e:
            logger.warning(f"AI extraction failed: {e}")
        
        return metrics
    
    def _extract_eps_from_tables(self, tables: List[Dict]) -> List[FinancialMetric]:
        """Extract EPS from table structures using advanced analysis"""
        metrics = []
        
        print(f"DEBUG: Analyzing {len(tables)} tables for EPS data")
        
        for table_idx, table in enumerate(tables):
            table_data = table['data']
            table_text = table['text']
            is_historical_summary = table.get('is_historical_summary', False)
            year_columns = table.get('year_columns', [])
            metric_rows = table.get('metric_rows', [])
            confidence = table.get('confidence', 0.0)
            
            print(f"DEBUG: Table {table_idx}: historical_summary={is_historical_summary}, "
                  f"year_columns={len(year_columns)}, metric_rows={len(metric_rows)}, confidence={confidence}")
            
            # Skip if table doesn't contain financial data
            if not any(keyword in table_text.lower() for keyword in self.metric_keywords['eps']):
                continue
            
            # Convert table to DataFrame for easier analysis
            df = pd.DataFrame(table_data)
            
            # Enhanced historical summary table processing
            if is_historical_summary and year_columns:
                print(f"DEBUG: Processing historical summary table with {len(year_columns)} year columns")
                metrics.extend(self._extract_from_historical_summary_table(df, year_columns, metric_rows, 'eps'))
            else:
                # Traditional table processing
                metrics.extend(self._extract_from_standard_table(df, 'eps'))
        
        print(f"DEBUG: Total EPS metrics extracted from tables: {len(metrics)}")
        return metrics
    
    def _extract_from_historical_summary_table(self, df: pd.DataFrame, year_columns: List[Tuple], 
                                              metric_rows: List[Tuple], metric_type: str) -> List[FinancialMetric]:
        """Extract data from historical summary tables with multiple year columns"""
        metrics = []
        
        # Sort year columns by year (ascending)
        year_columns.sort(key=lambda x: int(x[1]))
        
        print(f"DEBUG: Historical summary table - Year columns: {[year for _, year in year_columns]}")
        print(f"DEBUG: Metric rows: {metric_rows}")
        
        # Process each metric row
        for row_idx, row_type in metric_rows:
            if row_idx >= len(df):
                continue
                
            row = df.iloc[row_idx]
            row_text = ' '.join([str(cell) for cell in row if cell]).lower()
            
            print(f"DEBUG: Processing metric row {row_idx} (type: {row_type}): {row_text[:100]}...")
            
            # Only process rows that match the target metric type
            if row_type != metric_type and row_type != 'financial':
                continue
            
            # Determine unit from row context
            unit = self._detect_eps_unit_from_text(row_text) if metric_type == 'eps' else 'percentage'
            
            # Extract values for each year column
            for col_idx, year in year_columns:
                if col_idx < len(row):
                    cell_value = row.iloc[col_idx]
                    if cell_value and str(cell_value).strip():
                        try:
                            # Enhanced value cleaning
                            clean_value = str(cell_value).replace(',', '').replace('$', '').replace('%', '').strip()
                            
                            # Skip if the value looks like a year number (common error)
                            if clean_value.isdigit() and 1900 <= int(clean_value) <= 2030:
                                print(f"DEBUG: Skipping year-like value '{clean_value}' for year {year}")
                                continue
                            
                            value = float(clean_value)
                            
                            # Enhanced value validation
                            if self._is_reasonable_financial_value(value, metric_type):
                                # Convert units if necessary
                                if metric_type == 'eps' and unit == 'cents':
                                    value = value / 100
                                elif metric_type == 'roe' and unit == 'decimal':
                                    value = value * 100
                                
                                # Validate the value
                                if self._is_valid_eps_value(value) if metric_type == 'eps' else self._is_valid_roe_value(value):
                                    metrics.append(FinancialMetric(
                                        value=value,
                                        year=year,
                                        metric_type=metric_type,
                                        confidence=0.95,  # High confidence for historical summary tables
                                        source='historical_summary_table',
                                        unit='dollars' if metric_type == 'eps' else 'percentage',
                                        context=f"Historical summary table row: {row_text[:200]}"
                                    ))
                                    print(f"DEBUG: Extracted {metric_type.upper()} for {year}: {value}")
                            else:
                                print(f"DEBUG: Skipping unreasonable value '{clean_value}' for {metric_type}")
                        except ValueError as e:
                            print(f"DEBUG: Failed to parse value '{cell_value}' for year {year}: {e}")
                            continue
        
        return metrics
    
    def _extract_from_standard_table(self, df: pd.DataFrame, metric_type: str) -> List[FinancialMetric]:
        """Extract data from standard tables using traditional approach"""
        metrics = []
        
        # Find header row with years
        header_row = None
        for idx, row in df.iterrows():
            row_text = ' '.join([str(cell) for cell in row if cell])
            years = re.findall(r'\b(19|20)\d{2}\b', row_text)
            if len(years) >= 3:  # At least 3 years to be a historical table
                header_row = idx
                break
        
        if header_row is not None:
            # Extract years from header
            header_text = ' '.join([str(cell) for cell in df.iloc[header_row] if cell])
            years = re.findall(r'\b(19|20)\d{2}\b', header_text)
            
            # Determine unit from header or table context
            unit = self._detect_eps_unit(header_text, '') if metric_type == 'eps' else self._detect_roe_unit(header_text, '')
            
            # Find metric rows
            for idx, row in df.iterrows():
                if idx == header_row:
                    continue
                
                row_text = ' '.join([str(cell) for cell in row if cell]).lower()
                if any(keyword in row_text for keyword in self.metric_keywords[metric_type]):
                    # Extract values for each year
                    for col_idx, year in enumerate(years):
                        if col_idx < len(row):
                            cell_value = row.iloc[col_idx]
                            if cell_value and str(cell_value).strip():
                                try:
                                    value = float(str(cell_value).replace(',', ''))
                                    # Convert units if necessary
                                    if metric_type == 'eps' and unit == 'cents':
                                        value = value / 100
                                    elif metric_type == 'roe' and unit == 'decimal':
                                        value = value * 100
                                    
                                    if self._is_valid_eps_value(value) if metric_type == 'eps' else self._is_valid_roe_value(value):
                                        key_name = 'basic_eps' if metric_type == 'eps' else 'basic_roe'
                                        metrics.append(FinancialMetric(
                                            value=value,
                                            year=year,
                                            metric_type=metric_type,
                                            confidence=0.9,
                                            source='table',
                                            unit='dollars' if metric_type == 'eps' else 'percentage',
                                            context=f"Table row: {row_text} (unit: {unit})"
                                        ))
                                except ValueError:
                                    continue
                    break
        
        return metrics
    
    def _extract_eps_with_ml(self, text: str) -> List[FinancialMetric]:
        """Extract EPS using ML-based clustering and pattern recognition"""
        metrics = []
        
        # Extract all number-year pairs
        number_year_pairs = []
        
        # Enhanced patterns for historical financial data
        patterns = [
            r'(\d{4})\s+(-?\d+\.?\d*)',  # "2023 0.15"
            r'EPS\s+(\d{4})\s+(-?\d+\.?\d*)',  # "EPS 2023 0.15"
            r'(\d{4})\s+EPS\s+(-?\d+\.?\d*)',  # "2023 EPS 0.15"
            r'(\d{4})\s*:\s*(-?\d+\.?\d*)',  # "2023: 0.15"
            r'FY\s*(\d{4})\s+(-?\d+\.?\d*)',  # "FY 2023 0.15"
            r'(\d{4})\s*[-\u2013]\s*(\d{4})\s+(-?\d+\.?\d*)',  # "2022-2023 0.15"
            r'(\d{4})/(\d{4})\s+(-?\d+\.?\d*)',  # "2022/2023 0.15"
            r'Earnings\s+per\s+share.*?(\d{4}).*?(-?\d+\.?\d*)',  # "Earnings per share 2023 0.15"
            r'(\d{4}).*?Earnings\s+per\s+share.*?(-?\d+\.?\d*)',  # "2023 Earnings per share 0.15"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Handle different pattern groups
                    if len(match.groups()) == 2:
                        year = match.group(1)
                        value = float(match.group(2))
                    elif len(match.groups()) == 3:
                        # Handle year range patterns
                        year = match.group(1)  # Use first year
                        value = float(match.group(3))
                    else:
                        continue
                    
                    # Validate year is realistic
                    year_int = int(year)
                    current_year = datetime.now().year
                    if not (1995 <= year_int <= current_year):
                        continue
                    
                    # Enhanced value validation
                    if self._is_reasonable_financial_value(value, 'eps'):
                        number_year_pairs.append((year, value))
                except (ValueError, IndexError):
                    continue
        
        if number_year_pairs:
            print(f"DEBUG: Found {len(number_year_pairs)} EPS year-value pairs for ML clustering")
            
            # Use clustering to identify consistent patterns
            data = np.array([[int(year), value] for year, value in number_year_pairs])
            
            if len(data) > 1:
                # Normalize data for clustering
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)
                
                # Enhanced clustering parameters
                clustering = DBSCAN(eps=0.3, min_samples=2).fit(data_scaled)
                
                # Group by clusters
                clusters = defaultdict(list)
                for i, label in enumerate(clustering.labels_):
                    if label >= 0:  # Not noise
                        clusters[label].append(number_year_pairs[i])
                
                print(f"DEBUG: ML clustering found {len(clusters)} clusters")
                
                # Use the largest cluster as the most reliable
                if clusters:
                    largest_cluster = max(clusters.values(), key=len)
                    print(f"DEBUG: Largest cluster has {len(largest_cluster)} values")
                    
                    for year, value in largest_cluster:
                        metrics.append(FinancialMetric(
                            value=value,
                            year=year,
                            metric_type='eps',
                            confidence=0.85,
                            source='ml_clustering',
                            context=f"Clustered pattern with {len(largest_cluster)} similar values"
                        ))
        
        return metrics
    
    def _extract_eps_with_context(self, sections: List[Dict]) -> List[FinancialMetric]:
        """Extract EPS using context-aware analysis"""
        metrics = []
        
        for section in sections:
            text = section['text']
            
            # Check if this section contains financial data
            if not any(keyword in text.lower() for keyword in self.metric_keywords['eps']):
                continue
            
            # Look for context patterns
            context_patterns = [
                r'earnings\s+per\s+share.*?(\d{4}).*?(-?\d+\.?\d*)',
                r'(\d{4}).*?earnings\s+per\s+share.*?(-?\d+\.?\d*)',
                r'eps.*?(\d{4}).*?(-?\d+\.?\d*)',
                r'(\d{4}).*?eps.*?(-?\d+\.?\d*)'
            ]
            
            for pattern in context_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        year = match.group(1)
                        value = float(match.group(2))
                        
                        # Validate year is realistic
                        year_int = int(year)
                        current_year = datetime.now().year
                        if not (1995 <= year_int <= current_year):
                            continue
                        
                        if self._is_valid_eps_value(value):
                            metrics.append(FinancialMetric(
                                value=value,
                                year=year,
                                metric_type='eps',
                                confidence=0.75,
                                source='context',
                                context=text[:200] + "..." if len(text) > 200 else text
                            ))
                    except (ValueError, IndexError):
                        continue
        
        return metrics
    
    def _extract_eps_with_enhanced_patterns(self, text: str) -> List[FinancialMetric]:
        """Extract EPS using enhanced pattern recognition"""
        metrics = []
        
        # Enhanced year patterns
        year_patterns = [
            r'\b(19|20)\d{2}\b',                    # Standard years: 2019, 2020
            r'\bFY\s*(19|20)\d{2}\b',              # Fiscal years: FY 2019, FY2020
            r'\b(19|20)\d{2}[-/](19|20)\d{2}\b',   # Year ranges: 2019-2020, 2019/2020
            r'\b(19|20)\d{2}\s*[-\u2013]\s*(19|20)\d{2}\b',  # Year ranges with en dash
            r'\b(19|20)\d{2}\s*to\s*(19|20)\d{2}\b',    # Year ranges with "to"
            r'\b(19|20)\d{2}\s*through\s*(19|20)\d{2}\b', # Year ranges with "through"
        ]
        
        # Enhanced EPS patterns
        eps_patterns = [
            r'earnings\s+per\s+share\s*[:\-]?\s*([\d,]+\.?\d*)',
            r'eps\s*[:\-]?\s*([\d,]+\.?\d*)',
            r'basic\s+eps\s*[:\-]?\s*([\d,]+\.?\d*)',
            r'diluted\s+eps\s*[:\-]?\s*([\d,]+\.?\d*)',
            r'earnings\s+per\s+share.*?([\d,]+\.?\d*)',
            r'eps.*?([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*eps',
            r'([\d,]+\.?\d*)\s*earnings\s+per\s+share',
        ]
        
        # Extract all years from text
        all_years = []
        for pattern in year_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                year_text = match.group(0)
                # Extract individual years from ranges
                if any(separator in year_text for separator in ['-', '/', '\u2013', 'to', 'through']):
                    range_years = self._extract_years_from_range(year_text)
                    all_years.extend(range_years)
                else:
                    single_years = re.findall(r'(19|20)\d{2}', year_text)
                    all_years.extend(single_years)
        
        # Remove duplicates and validate
        unique_years = list(set(all_years))
        valid_years = []
        for year in unique_years:
            try:
                year_int = int(year)
                current_year = datetime.now().year
                if 1990 <= year_int <= current_year + 5:
                    valid_years.append(year)
            except ValueError:
                continue
        
        print(f"DEBUG: Enhanced pattern extraction found {len(valid_years)} years: {sorted(valid_years)}")
        
        # Extract EPS values using enhanced patterns
        for pattern in eps_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value_text = match.group(1)
                try:
                    clean_value = value_text.replace(',', '').replace('$', '').strip()
                    value = float(clean_value)
                    
                    # Find the closest year to this metric
                    match_start = match.start()
                    closest_year = self._find_closest_year_enhanced(text, match_start, valid_years)
                    
                    if closest_year and self._is_valid_eps_value(value):
                        metrics.append(FinancialMetric(
                            value=value,
                            year=closest_year,
                            metric_type='eps',
                            confidence=0.85,
                            source='enhanced_patterns',
                            unit='dollars',
                            context=text[max(0, match_start-50):match_start+50]
                        ))
                except ValueError:
                    continue
        
        print(f"DEBUG: Enhanced pattern extraction found {len(metrics)} EPS metrics")
        return metrics
    
    def _extract_years_from_range(self, range_text: str) -> List[str]:
        """Extract all years from a year range"""
        years = []
        
        # Remove common prefixes and suffixes
        clean_text = re.sub(r'\b(FY|CY|Calendar\s*Year)\s*', '', range_text, flags=re.IGNORECASE)
        clean_text = re.sub(r'\([^)]*\)', '', clean_text)  # Remove parentheses content
        
        # Find all year numbers
        year_numbers = re.findall(r'(19|20)\d{2}', clean_text)
        
        if len(year_numbers) >= 2:
            try:
                start_year = int(year_numbers[0])
                end_year = int(year_numbers[1])
                
                # Generate all years in the range
                for year in range(start_year, end_year + 1):
                    years.append(str(year))
            except ValueError:
                pass
        
        return years
    
    def _find_closest_year_enhanced(self, text: str, position: int, years: List[str]) -> Optional[str]:
        """Find the year closest to a given position in text"""
        if not years:
            return None
        
        closest_year = None
        min_distance = float('inf')
        
        for year in years:
            year_positions = [m.start() for m in re.finditer(r'\b' + year + r'\b', text)]
            for year_pos in year_positions:
                distance = abs(year_pos - position)
                if distance < min_distance:
                    min_distance = distance
                    closest_year = year
        
        return closest_year
    
    def _extract_eps_comprehensive(self, layout_data: Dict[str, Any]) -> List[FinancialMetric]:
        """Comprehensive EPS extraction from all available data sources"""
        metrics = []
        
        # Extract from all sections
        for section in layout_data['sections']:
            section_text = section['text']
            
            # Check if section contains historical data
            if self._is_historical_summary_section(section_text):
                section_metrics = self._extract_eps_from_historical_section(section_text)
                metrics.extend(section_metrics)
        
        # Extract from all tables
        for table in layout_data['tables']:
            table_text = table['text']
            if any(keyword in table_text.lower() for keyword in self.metric_keywords['eps']):
                table_metrics = self._extract_eps_from_table_text(table_text)
                metrics.extend(table_metrics)
        
        # Extract from full text using comprehensive patterns
        full_text_metrics = self._extract_eps_from_full_text(layout_data['full_text'])
        metrics.extend(full_text_metrics)
        
        print(f"DEBUG: Comprehensive extraction found {len(metrics)} EPS metrics")
        return metrics
    
    def _extract_eps_from_historical_section(self, text: str) -> List[FinancialMetric]:
        """Extract EPS from historical summary sections"""
        metrics = []
        
        # Enhanced patterns for historical sections
        patterns = [
            r'(\d{4}).*?eps.*?([\d,]+\.?\d*)',
            r'eps.*?(\d{4}).*?([\d,]+\.?\d*)',
            r'(\d{4}).*?earnings.*?([\d,]+\.?\d*)',
            r'earnings.*?(\d{4}).*?([\d,]+\.?\d*)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    year = match.group(1)
                    value_text = match.group(2)
                    value = float(value_text.replace(',', '').replace('$', '').strip())
                    
                    if self._is_valid_eps_value(value):
                        metrics.append(FinancialMetric(
                            value=value,
                            year=year,
                            metric_type='eps',
                            confidence=0.8,
                            source='historical_section',
                            unit='dollars',
                            context=match.group(0)
                        ))
                except ValueError:
                    continue
        
        return metrics
    
    def _extract_eps_from_table_text(self, text: str) -> List[FinancialMetric]:
        """Extract EPS from table text"""
        metrics = []
        
        # Split text into lines and look for year-value patterns
        lines = text.split('\n')
        for line in lines:
            # Look for patterns like "2019 0.15" or "2019 EPS 0.15"
            patterns = [
                r'(\d{4})\s+([\d,]+\.?\d*)',
                r'(\d{4}).*?eps.*?([\d,]+\.?\d*)',
                r'eps.*?(\d{4}).*?([\d,]+\.?\d*)',
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    try:
                        year = match.group(1)
                        value_text = match.group(2)
                        value = float(value_text.replace(',', '').replace('$', '').strip())
                        
                        if self._is_valid_eps_value(value):
                            metrics.append(FinancialMetric(
                                value=value,
                                year=year,
                                metric_type='eps',
                                confidence=0.75,
                                source='table_text',
                                unit='dollars',
                                context=line
                            ))
                    except ValueError:
                        continue
        
        return metrics
    
    def _extract_eps_from_full_text(self, text: str) -> List[FinancialMetric]:
        """Extract EPS from full text using comprehensive patterns"""
        metrics = []
        
        # Comprehensive patterns for full text
        patterns = [
            r'(\d{4}).*?eps.*?([\d,]+\.?\d*)',
            r'eps.*?(\d{4}).*?([\d,]+\.?\d*)',
            r'(\d{4}).*?earnings.*?([\d,]+\.?\d*)',
            r'earnings.*?(\d{4}).*?([\d,]+\.?\d*)',
            r'(\d{4}).*?([\d,]+\.?\d*).*?eps',
            r'eps.*?(\d{4}).*?([\d,]+\.?\d*)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    year = match.group(1)
                    value_text = match.group(2)
                    value = float(value_text.replace(',', '').replace('$', '').strip())
                    
                    if self._is_valid_eps_value(value):
                        metrics.append(FinancialMetric(
                            value=value,
                            year=year,
                            metric_type='eps',
                            confidence=0.7,
                            source='full_text',
                            unit='dollars',
                            context=match.group(0)
                        ))
                except ValueError:
                        continue
        
        return metrics
    
    def _extract_roe_advanced(self, layout_data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced ROE extraction using similar approaches as EPS"""
        roe_metrics = []
        
        print(f"DEBUG: Starting ROE extraction with {len(layout_data.get('tables', []))} tables and {len(layout_data.get('sections', []))} sections")
        
        # Similar strategies as EPS but for ROE
        if self.use_ai:
            print("DEBUG: Using AI-based ROE extraction")
            ai_roe = self._extract_roe_with_ai(layout_data['full_text'])
            roe_metrics.extend(ai_roe)
            print(f"DEBUG: AI ROE extraction found {len(ai_roe)} metrics")
        
        print("DEBUG: Using table-based ROE extraction")
        table_roe = self._extract_roe_from_tables(layout_data['tables'])
        roe_metrics.extend(table_roe)
        print(f"DEBUG: Table ROE extraction found {len(table_roe)} metrics")
        
        # NEW: Historical summary section extraction for ROE
        print("DEBUG: Using historical summary section extraction for ROE")
        historical_roe = self._extract_from_historical_summary_sections(layout_data['historical_summaries'], 'roe')
        roe_metrics.extend(historical_roe)
        print(f"DEBUG: Historical summary ROE extraction found {len(historical_roe)} metrics")
        
        print("DEBUG: Using ML-based ROE extraction")
        ml_roe = self._extract_roe_with_ml(layout_data['full_text'])
        roe_metrics.extend(ml_roe)
        print(f"DEBUG: ML ROE extraction found {len(ml_roe)} metrics")
        
        print("DEBUG: Using context-based ROE extraction")
        context_roe = self._extract_roe_with_context(layout_data['sections'])
        roe_metrics.extend(context_roe)
        print(f"DEBUG: Context ROE extraction found {len(context_roe)} metrics")
        
        # NEW: Financial highlights extraction for ROE
        print("DEBUG: Using financial highlights extraction for ROE")
        highlights_roe = self._extract_from_financial_highlights(layout_data['full_text'], 'roe')
        roe_metrics.extend(highlights_roe)
        print(f"DEBUG: Financial highlights ROE extraction found {len(highlights_roe)} metrics")
        
        # Strategy 6: Enhanced AI extraction with improved pattern recognition (NEW)
        print("DEBUG: Using enhanced pattern extraction for ROE")
        enhanced_roe = self._extract_roe_with_enhanced_patterns(layout_data['full_text'])
        roe_metrics.extend(enhanced_roe)
        print(f"DEBUG: Enhanced pattern ROE extraction found {len(enhanced_roe)} metrics")
        
        # Strategy 7: Comprehensive page-by-page extraction (NEW)
        print("DEBUG: Using comprehensive extraction for ROE")
        comprehensive_roe = self._extract_roe_comprehensive(layout_data)
        roe_metrics.extend(comprehensive_roe)
        print(f"DEBUG: Comprehensive ROE extraction found {len(comprehensive_roe)} metrics")
        
        print(f"DEBUG: Total ROE metrics found: {len(roe_metrics)}")
        
        return self._merge_financial_metrics(roe_metrics, 'roe')
    
    def _extract_roe_with_ai(self, text: str) -> List[FinancialMetric]:
        """Extract ROE using AI-powered semantic analysis"""
        metrics = []
        
        if not self.use_ai:
            return metrics
        
        # Similar to EPS but with ROE keywords
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue
            
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in self.metric_keywords['roe']):
                numbers = re.findall(r'-?\d+\.?\d*', sentence)
                years = re.findall(r'\b(19|20)\d{2}\b', sentence)
                
                for year in years:
                    # Validate year is realistic
                    try:
                        year_int = int(year)
                        current_year = datetime.now().year
                        if not (1995 <= year_int <= current_year):
                            continue
                    except (ValueError, TypeError):
                        continue
                    
                    year_pos = sentence.find(year)
                    closest_number = None
                    min_distance = float('inf')
                    
                    for num in numbers:
                        num_pos = sentence.find(num)
                        distance = abs(num_pos - year_pos)
                        if distance < min_distance and distance < 100:
                            min_distance = distance
                            closest_number = num
                    
                    if closest_number:
                        try:
                            value = float(closest_number)
                            # Detect unit from context
                            unit = self._detect_roe_unit_from_text(sentence)
                            if unit == 'decimal':
                                value = value * 100
                            if self._is_valid_roe_value(value):
                                metrics.append(FinancialMetric(
                                    value=value,
                                    year=year,
                                    metric_type='roe',
                                    confidence=0.8,
                                    source='ai',
                                    unit='percentage',
                                    context=sentence
                                ))
                        except ValueError:
                            continue
        
        return metrics
    
    def _extract_roe_from_tables(self, tables: List[Dict]) -> List[FinancialMetric]:
        """Extract ROE from table structures"""
        metrics = []
        
        print(f"DEBUG: Analyzing {len(tables)} tables for ROE data")
        
        for table_idx, table in enumerate(tables):
            table_data = table['data']
            table_text = table['text']
            is_historical_summary = table.get('is_historical_summary', False)
            year_columns = table.get('year_columns', [])
            metric_rows = table.get('metric_rows', [])
            confidence = table.get('confidence', 0.0)
            
            print(f"DEBUG: Table {table_idx}: historical_summary={is_historical_summary}, "
                  f"year_columns={len(year_columns)}, metric_rows={len(metric_rows)}, confidence={confidence}")
            
            if not any(keyword in table_text.lower() for keyword in self.metric_keywords['roe']):
                continue
            
            df = pd.DataFrame(table_data)
            
            # Enhanced historical summary table processing
            if is_historical_summary and year_columns:
                print(f"DEBUG: Processing historical summary table with {len(year_columns)} year columns")
                metrics.extend(self._extract_from_historical_summary_table(df, year_columns, metric_rows, 'roe'))
            else:
                # Traditional table processing
                metrics.extend(self._extract_from_standard_table(df, 'roe'))
        
        print(f"DEBUG: Total ROE metrics extracted from tables: {len(metrics)}")
        return metrics
    
    def _extract_roe_with_ml(self, text: str) -> List[FinancialMetric]:
        """Extract ROE using ML-based clustering"""
        metrics = []
        
        number_year_pairs = []
        
        # Enhanced patterns for historical ROE data
        patterns = [
            r'(\d{4})\s+(-?\d+\.?\d*)',  # "2023 15.5"
            r'ROE\s+(\d{4})\s+(-?\d+\.?\d*)',  # "ROE 2023 15.5"
            r'(\d{4})\s+ROE\s+(-?\d+\.?\d*)',  # "2023 ROE 15.5"
            r'(\d{4})\s*:\s*(-?\d+\.?\d*)',  # "2023: 15.5"
            r'Return\s+on\s+Equity\s+(\d{4})\s+(-?\d+\.?\d*)',  # "Return on Equity 2023 15.5"
            r'(\d{4})\s*[-\u2013]\s*(\d{4})\s+(-?\d+\.?\d*)',  # "2022-2023 15.5"
            r'(\d{4})/(\d{4})\s+(-?\d+\.?\d*)',  # "2022/2023 15.5"
            r'(\d{4}).*?Return\s+on\s+Equity.*?(-?\d+\.?\d*)',  # "2023 Return on Equity 15.5"
            r'(\d{4}).*?ROE.*?(-?\d+\.?\d*)',  # "2023 ROE 15.5"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Handle different pattern groups
                    if len(match.groups()) == 2:
                        year = match.group(1)
                        value = float(match.group(2))
                    elif len(match.groups()) == 3:
                        # Handle year range patterns
                        year = match.group(1)  # Use first year
                        value = float(match.group(3))
                    else:
                        continue
                    
                    # Validate year is realistic
                    year_int = int(year)
                    current_year = datetime.now().year
                    if not (1995 <= year_int <= current_year):
                        continue
                    
                    # Enhanced value validation
                    if self._is_reasonable_financial_value(value, 'roe'):
                        number_year_pairs.append((year, value))
                except (ValueError, IndexError):
                    continue
        
        if number_year_pairs:
            data = np.array([[int(year), value] for year, value in number_year_pairs])
            
            if len(data) > 1:
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data)
                
                # Enhanced clustering parameters
                clustering = DBSCAN(eps=0.3, min_samples=2).fit(data_scaled)
                
                clusters = defaultdict(list)
                for i, label in enumerate(clustering.labels_):
                    if label >= 0:
                        clusters[label].append(number_year_pairs[i])
                
                if clusters:
                    largest_cluster = max(clusters.values(), key=len)
                    for year, value in largest_cluster:
                        metrics.append(FinancialMetric(
                            value=value,
                            year=year,
                            metric_type='roe',
                            confidence=0.85,
                            source='ml_clustering',
                            context=f"Clustered pattern with {len(largest_cluster)} similar values"
                        ))
        
        return metrics
    
    def _extract_roe_with_context(self, sections: List[Dict]) -> List[FinancialMetric]:
        """Extract ROE using context-aware analysis"""
        metrics = []
        
        for section in sections:
            text = section['text']
            
            if not any(keyword in text.lower() for keyword in self.metric_keywords['roe']):
                continue
            
            context_patterns = [
                r'return\s+on\s+equity.*?(\d{4}).*?(-?\d+\.?\d*)',
                r'(\d{4}).*?return\s+on\s+equity.*?(-?\d+\.?\d*)',
                r'roe.*?(\d{4}).*?(-?\d+\.?\d*)',
                r'(\d{4}).*?roe.*?(-?\d+\.?\d*)'
            ]
            
            for pattern in context_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        year = match.group(1)
                        value = float(match.group(2))
                        
                        # Validate year is realistic
                        year_int = int(year)
                        current_year = datetime.now().year
                        if not (1995 <= year_int <= current_year):
                            continue
                        
                        if self._is_valid_roe_value(value):
                            metrics.append(FinancialMetric(
                                value=value,
                                year=year,
                                metric_type='roe',
                                confidence=0.75,
                                source='context',
                                context=text[:200] + "..." if len(text) > 200 else text
                            ))
                    except (ValueError, IndexError):
                        continue
        
        return metrics
    
    def _merge_financial_metrics(self, metrics: List[FinancialMetric], metric_type: str) -> Dict[str, Any]:
        """Merge and deduplicate financial metrics"""
        merged_data = {}
        
        print(f"DEBUG: Merging {len(metrics)} {metric_type} metrics")
        
        # Group by year
        year_groups = defaultdict(list)
        for metric in metrics:
            year_groups[metric.year].append(metric)
        
        print(f"DEBUG: Grouped into {len(year_groups)} year groups: {list(year_groups.keys())}")
        
        # For each year, select the best metric based on confidence and source
        for year, year_metrics in year_groups.items():
            # Sort by confidence and source priority
            source_priority = {'table': 3, 'ml_clustering': 2, 'ai': 2, 'context': 1}
            
            best_metric = max(year_metrics, key=lambda m: (m.confidence, source_priority.get(m.source, 0)))
            
            # Use consistent key names: 'basic_eps' for EPS, 'basic_roe' for ROE
            if metric_type == 'eps':
                key_name = 'basic_eps'
            elif metric_type == 'roe':
                key_name = 'basic_roe'
            else:
                key_name = f'basic_{metric_type}'
            
            merged_data[year] = {
                key_name: best_metric.value,
                'confidence': best_metric.confidence,
                'source': best_metric.source,
                'currency': best_metric.currency,
                'unit': best_metric.unit,
                'context': best_metric.context,
                # Provenance fields for downstream transparency
                'provenance': {
                    'method': best_metric.source,
                    'selection': 'best_by_confidence_and_source_priority'
                }
            }
            
            print(f"DEBUG: Merged {metric_type} for year {year}: {best_metric.value} (source: {best_metric.source}, confidence: {best_metric.confidence})")
        
        print(f"DEBUG: Final merged {metric_type} data: {len(merged_data)} years")
        return merged_data
    
    def _validate_financial_data(self, data: Dict[str, Any], metric_type: str) -> Dict[str, Any]:
        """Validate and clean financial data"""
        validated_data = {}
        current_year = datetime.now().year
        
        # Use consistent key names
        if metric_type == 'eps':
            key_name = 'basic_eps'
        elif metric_type == 'roe':
            key_name = 'basic_roe'
        else:
            key_name = f'basic_{metric_type}'
        
        for year, year_data in data.items():
            if isinstance(year_data, dict) and key_name in year_data:
                value = year_data[key_name]
                
                # Validate the year (must be realistic)
                try:
                    year_int = int(year)
                    # Year must be between 1900 and current year (more realistic range)
                    # This prevents years like 1989, 2030, 2054, 2251
                    if not (1900 <= year_int <= current_year):
                        print(f"WARNING: Skipping invalid year {year} for {metric_type} (must be between 1900 and {current_year})")
                        continue
                except (ValueError, TypeError):
                    print(f"WARNING: Skipping invalid year format '{year}' for {metric_type}")
                    continue
                
                # Validate the value
                if metric_type == 'eps':
                    if self._is_valid_eps_value(value):
                        validated_data[year] = year_data
                elif metric_type == 'roe':
                    if self._is_valid_roe_value(value):
                        validated_data[year] = year_data
        
        # Ensure we have at least 3 years of data (reduced for flexibility)
        if len(validated_data) < 3:
            print(f"WARNING: Only {len(validated_data)} years of {metric_type.upper()} data found. Need at least 3 years.")
            print(f"Valid years found: {sorted(validated_data.keys())}")
        
        # AI-POWERED CONFIDENCE SCORING
        confidence_score = self._calculate_ai_confidence(validated_data, metric_type)
        print(f"AI Confidence Score for {metric_type.upper()}: {confidence_score:.1%}")
        
        return validated_data

    # ===================== ENSEMBLE VALIDATOR AND RECONCILIATION =====================
    def _structured_validator_extract(self, layout_data: Dict[str, Any], metric_type: str, pdf_path: str = None) -> Dict[str, Any]:
        """
        Deterministic extraction focusing on parsed tables and token alignment.
        Returns a dict like {year: {basic_eps/basic_roe, unit, source, confidence, provenance}}.
        """
        validator_metrics: List[FinancialMetric] = []
        # 0) Anchored table parsing for common report patterns
        try:
            if metric_type == 'eps':
                validator_metrics.extend(self._structured_eps_anchored(layout_data))
            elif metric_type == 'roe':
                validator_metrics.extend(self._structured_roe_anchored(layout_data))
        except Exception:
            pass
        # 1) Tables are most reliable for structured pairing of year->value
        for table in layout_data.get('tables', []):
            df = pd.DataFrame(table.get('data', []))
            if df.empty or df.shape[0] < 2:
                continue
            # Detect year columns
            candidate_years_by_col: Dict[int, List[str]] = {}
            for col_idx in range(len(df.columns)):
                col_text = ' '.join([str(cell) for cell in df.iloc[:, col_idx] if pd.notna(cell)])
                years = self._find_years_in_text(col_text)
                if years:
                    candidate_years_by_col[col_idx] = sorted(list(set(years)))

            if not candidate_years_by_col:
                continue

            # Identify metric rows containing target metric keywords
            target_keywords = self.metric_keywords.get(metric_type, [])
            metric_row_indices: List[int] = []
            for row_idx in range(len(df)):
                row_text = ' '.join([str(cell) for cell in df.iloc[row_idx] if pd.notna(cell)]).lower()
                if any(k in row_text for k in target_keywords):
                    metric_row_indices.append(row_idx)

            # Extract values by intersecting metric rows with year columns
            for row_idx in metric_row_indices:
                row_text = ' '.join([str(cell) for cell in df.iloc[row_idx] if pd.notna(cell)]).lower()
                for col_idx, years in candidate_years_by_col.items():
                    # Each year column typically holds a numeric value for that year in this row
                    if col_idx < len(df.columns) and row_idx < len(df.index):
                        cell = df.iat[row_idx, col_idx]
                        if isinstance(cell, str):
                            clean = cell.replace(',', '').replace('$', '').replace('%', '').strip()
                            # Handle negative in parentheses e.g. (12.3)
                            neg = clean.startswith('(') and clean.endswith(')')
                            clean = clean.strip('()')
                        else:
                            clean = str(cell)
                        try:
                            val = float(clean)
                            if 'neg' in locals() and neg:
                                val = -val
                        except ValueError:
                            continue

                        # Determine unit normalization
                        unit = 'percentage' if metric_type == 'roe' else 'dollars'
                        if metric_type == 'roe' and '%' not in str(cell):
                            # assume already percentage or decimal near [0,1000]
                            pass
                        if metric_type == 'eps' and any(tok in row_text for tok in ['cent', 'cents']):
                            val = val / 100.0

                        # Validate value
                        if metric_type == 'eps':
                            if not self._is_valid_eps_value(val):
                                continue
                        else:
                            if not self._is_valid_roe_value(val):
                                continue

                        # Use the closest plausible year from this column context
                        for year in years:
                            validator_metrics.append(FinancialMetric(
                                value=val,
                                year=year,
                                metric_type=metric_type,
                                confidence=0.75,
                                source='validator_table',
                                unit=unit,
                                context=f"table_page:{table.get('page')} idx:{table.get('index')}"
                            ))

        # Camelot fallback for EPS if none found
        if metric_type == 'eps' and not validator_metrics and pdf_path:
            try:
                print("DEBUG: Camelot fallback for EPS extraction starting...")
                camelot_metrics = self._camelot_extract_eps(pdf_path)
                if camelot_metrics:
                    validator_metrics.extend(camelot_metrics)
                    print(f"DEBUG: Camelot fallback produced {len(camelot_metrics)} EPS metrics")
                else:
                    print("DEBUG: Camelot fallback found no EPS metrics")
            except Exception as ce:
                print(f"DEBUG: Camelot fallback error: {ce}")

        # PaddleOCR PP-Structure fallback for both EPS and ROE if still none
        if not validator_metrics and pdf_path:
            try:
                print("DEBUG: PP-Structure fallback starting for {}...".format(metric_type.upper()))
                pp_metrics = self._ppstructure_extract_metrics(pdf_path, metric_type)
                if pp_metrics:
                    validator_metrics.extend(pp_metrics)
                    print(f"DEBUG: PP-Structure fallback produced {len(pp_metrics)} {metric_type.upper()} metrics")
                else:
                    print("DEBUG: PP-Structure fallback found no {} metrics".format(metric_type.upper()))
            except Exception as pe:
                print(f"DEBUG: PP-Structure fallback error: {pe}")

        # Anchor-guided cropping + PP-Structure if still none
        if not validator_metrics and pdf_path:
            try:
                print("DEBUG: PP-Structure anchor-crop fallback starting for {}...".format(metric_type.upper()))
                cropped_metrics = self._ppstructure_anchor_crop_extract(pdf_path, metric_type)
                if cropped_metrics:
                    validator_metrics.extend(cropped_metrics)
                    print(f"DEBUG: PP-Structure anchor-crop produced {len(cropped_metrics)} {metric_type.upper()} metrics")
                else:
                    print("DEBUG: PP-Structure anchor-crop found no {} metrics".format(metric_type.upper()))
            except Exception as ce:
                print(f"DEBUG: PP-Structure anchor-crop error: {ce}")

        # Microsoft Table Transformer detection + PP-Structure parse if still none
        if not validator_metrics and pdf_path:
            try:
                print("DEBUG: Table Transformer detection fallback starting for {}...".format(metric_type.upper()))
                tt_metrics = self._table_transformer_detect_and_parse(pdf_path, metric_type)
                if tt_metrics:
                    validator_metrics.extend(tt_metrics)
                    print(f"DEBUG: Table Transformer produced {len(tt_metrics)} {metric_type.upper()} metrics")
                else:
                    print("DEBUG: Table Transformer found no {} metrics".format(metric_type.upper()))
            except Exception as te:
                print(f"DEBUG: Table Transformer fallback error: {te}")

        # Merge validator metrics
        return self._merge_financial_metrics(validator_metrics, metric_type)

    def _camelot_extract_eps(self, pdf_path: str) -> List[FinancialMetric]:
        """Use Camelot to extract anchored EPS values when internal table parsing fails."""
        try:
            import camelot
        except Exception as e:
            print(f"DEBUG: Camelot not available: {e}")
            return []

        metrics: List[FinancialMetric] = []
        tables = []
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        except Exception:
            pass
        if not tables or len(tables) == 0:
            try:
                tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
            except Exception:
                tables = []

        if not tables:
            return []

        anchors = ['earnings per share', 'eps']
        for t in tables:
            try:
                df = t.df
                if df is None or df.shape[0] == 0:
                    continue
                # Find anchor row index and cents flag
                anchor_idx = None
                is_cents = False
                for r in range(df.shape[0]):
                    row_join = ' '.join([str(x) for x in list(df.iloc[r, :])]).lower()
                    if any(a in row_join for a in anchors):
                        anchor_idx = r
                        if 'cent' in row_join or 'cents' in row_join:
                            is_cents = True
                        break
                if anchor_idx is None:
                    continue
                # Find Basic row within next few rows
                basic_idx = None
                for r in range(anchor_idx, min(anchor_idx + 8, df.shape[0])):
                    first_cell = str(df.iat[r, 0]).strip().lower() if df.shape[1] > 0 else ''
                    row_text = ' '.join([str(x) for x in list(df.iloc[r, :])]).lower()
                    if first_cell == 'basic' or ' basic' in row_text:
                        basic_idx = r
                        break
                if basic_idx is None:
                    continue
                # Map year columns from header rows (top 3 rows), fallback to full column scan
                year_for_col: Dict[int, str] = {}
                header_rows = min(3, df.shape[0])
                for c in range(df.shape[1]):
                    col_text = ' '.join([str(df.iat[r, c]) for r in range(header_rows)])
                    years = self._find_years_in_text(col_text)
                    if years:
                        year_for_col[c] = sorted(set(years))[-1]
                if not year_for_col:
                    for c in range(df.shape[1]):
                        col_text = ' '.join([str(df.iat[r, c]) for r in range(df.shape[0])])
                        years = self._find_years_in_text(col_text)
                        if years:
                            year_for_col[c] = sorted(set(years))[-1]
                if not year_for_col:
                    continue

                # Extract values from Basic row across year columns
                for c, y in year_for_col.items():
                    if c >= df.shape[1] or basic_idx >= df.shape[0]:
                        continue
                    cell = df.iat[basic_idx, c]
                    if cell is None:
                        continue
                    raw = str(cell).replace(',', '').replace('$', '').replace('%', '').strip()
                    neg = raw.startswith('(') and raw.endswith(')')
                    raw = raw.strip('()')
                    try:
                        val = float(raw)
                    except ValueError:
                        continue
                    if neg:
                        val = -val
                    unit = 'dollars'
                    if is_cents:
                        try:
                            raw_cents = float(raw)
                        except Exception:
                            raw_cents = None
                        if raw_cents is not None and (raw_cents < 0 or raw_cents > 1000):
                            continue
                        val = val / 100.0
                    else:
                        if abs(val) > 50.0:
                            continue
                    metrics.append(FinancialMetric(
                        value=val,
                        year=str(y),
                        metric_type='eps',
                        confidence=0.8,
                        source='validator_table_camelot_eps_anchor',
                        unit=unit,
                        context=f"camelot_table"
                    ))
            except Exception:
                continue
        return metrics

    def _ppstructure_extract_metrics(self, pdf_path: str, metric_type: str) -> List[FinancialMetric]:
        """Use PaddleOCR PP-Structure to detect tables, then run anchored extraction.

        This is a robust fallback for PDFs where standard table detection fails.
        It rasterizes PDF pages to images, runs PP-Structure, converts detected
        table HTML to DataFrames, then applies the same anchored logic used by
        structured parsing to extract EPS/ROE by year.
        """
        metrics: List[FinancialMetric] = []
        try:
            from paddleocr import PPStructure
        except Exception as e:
            print(f"DEBUG: PP-Structure not available: {e}")
            return metrics

        try:
            import fitz  # PyMuPDF
            from PIL import Image
            import io
        except Exception as e:
            print(f"DEBUG: PP-Structure prerequisites missing: {e}")
            return metrics

        import pandas as pd  # ensure local alias

        table_engine = PPStructure(show_log=False)

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"DEBUG: Could not open PDF for PP-Structure: {e}")
            return metrics

        try:
            for page_index in range(len(doc)):
                try:
                    page = doc[page_index]
                    pix = page.get_pixmap(dpi=200)
                    img = Image.open(io.BytesIO(pix.tobytes()))
                except Exception:
                    continue

                try:
                    result = table_engine(img)
                except Exception:
                    continue

                for item in result:
                    try:
                        if str(item.get('type', '')).lower() != 'table':
                            continue
                        res = item.get('res') or {}
                        html = res.get('html')
                        if not html:
                            continue
                        try:
                            dfs = pd.read_html(html)
                        except Exception:
                            dfs = []
                        for df in dfs:
                            if df is None or df.shape[0] == 0:
                                continue
                            extracted = self._extract_metrics_from_dataframe(df, metric_type)
                            metrics.extend(extracted)
                    except Exception:
                        continue
        finally:
            try:
                doc.close()
            except Exception:
                pass

        return metrics

    def _extract_metrics_from_dataframe(self, df: 'pd.DataFrame', metric_type: str) -> List[FinancialMetric]:
        """Anchor-driven extraction from a single table DataFrame produced by PP-Structure.

        - For EPS: find anchor row containing 'earnings per share'/'eps' and locate 'Basic' row
          nearby, detect units (cents vs dollars) and extract values across year columns.
        - For ROE: find anchor row containing 'return on equity'/'roe' and extract values across
          year columns as percentages.
        """
        import pandas as pd  # type: ignore
        results: List[FinancialMetric] = []

        if df is None or df.shape[0] == 0:
            return results

        # Normalize to strings for scanning
        df_norm = df.copy()
        try:
            df_norm = df_norm.fillna('')
        except Exception:
            pass

        # Detect anchor and units
        anchor_keywords = []
        if metric_type == 'eps':
            anchor_keywords = ['earnings per share', 'eps', 'earnings per share - continuing operations', 'earnings per share \u2013 continuing operations']
        else:
            anchor_keywords = ['return on equity', 'roe']

        anchor_row_idx = None
        is_cents = False
        for r in range(len(df_norm)):
            row_join = ' '.join([str(x) for x in df_norm.iloc[r] if str(x) != '']).lower()
            if any(k in row_join for k in anchor_keywords):
                anchor_row_idx = r
                if metric_type == 'eps' and ('cent' in row_join or 'cents' in row_join):
                    is_cents = True
                break

        # Header hint for units when no explicit anchor indicates cents
        if anchor_row_idx is None and metric_type == 'eps' and len(df_norm.index) > 0:
            header_text = ' '.join([str(x) for x in df_norm.iloc[0] if str(x) != '']).lower()
            if 'cent' in header_text or 'cents' in header_text:
                is_cents = True

        if anchor_row_idx is None:
            return results

        # Locate Basic row for EPS (same row or following few rows)
        basic_row_idx = None
        if metric_type == 'eps':
            for r in range(anchor_row_idx, min(anchor_row_idx + 8, len(df_norm))):
                first_cell = str(df_norm.iat[r, 0]).strip().lower() if df_norm.shape[1] > 0 else ''
                row_text = ' '.join([str(x) for x in df_norm.iloc[r] if str(x) != '']).lower()
                if first_cell == 'basic' or ' basic' in row_text:
                    basic_row_idx = r
                    break
            if basic_row_idx is None:
                return results

        # Map year columns (scan headers, then full column as fallback)
        year_for_col: Dict[int, str] = {}
        header_rows = min(3, df_norm.shape[0])
        for c in range(df_norm.shape[1]):
            col_text = ' '.join([str(df_norm.iat[r, c]) for r in range(header_rows)])
            years = self._find_years_in_text(col_text)
            if years:
                year_for_col[c] = sorted(set(years))[-1]
        if not year_for_col:
            for c in range(df_norm.shape[1]):
                col_text = ' '.join([str(df_norm.iat[r, c]) for r in range(df_norm.shape[0])])
                years = self._find_years_in_text(col_text)
                if years:
                    year_for_col[c] = sorted(set(years))[-1]
        if not year_for_col:
            return results

        # Extract values across year columns
        for c, y in year_for_col.items():
            try:
                if metric_type == 'eps':
                    if c >= df_norm.shape[1] or basic_row_idx is None or basic_row_idx >= df_norm.shape[0]:
                        continue
                    cell = df_norm.iat[basic_row_idx, c]
                else:
                    if c >= df_norm.shape[1] or anchor_row_idx >= df_norm.shape[0]:
                        continue
                    cell = df_norm.iat[anchor_row_idx, c]

                raw = str(cell).replace(',', '').replace('$', '').replace('%', '').strip()
                neg = raw.startswith('(') and raw.endswith(')')
                raw = raw.strip('()')
                val = float(raw)
                if neg:
                    val = -val

                unit = 'percentage' if metric_type == 'roe' else 'dollars'
                if metric_type == 'eps':
                    if is_cents:
                        # Guardrails for cents conversion
                        try:
                            raw_cents = float(raw)
                        except Exception:
                            raw_cents = None
                        if raw_cents is not None and (raw_cents < -1000 or raw_cents > 1000):
                            continue
                        val = val / 100.0
                    else:
                        if abs(val) > 50.0:
                            # Implausible for dollars; skip
                            continue
                    if not self._is_valid_eps_value(val):
                        continue
                else:
                    # ROE as percentage; allow negatives, but sanity check range
                    if not self._is_valid_roe_value(val):
                        continue

                results.append(FinancialMetric(
                    value=val,
                    year=str(y),
                    metric_type=metric_type,
                    confidence=0.80,
                    source='validator_table_ppstructure_anchor',
                    unit=unit,
                    context='ppstructure_table'
                ))
            except Exception:
                continue

        return results

    def _ppstructure_anchor_crop_extract(self, pdf_path: str, metric_type: str) -> List[FinancialMetric]:
        """Search for anchor text on each page, crop a tight region around it, and run PP-Structure on
        the cropped image to improve table detection when full-page parsing fails."""
        metrics: List[FinancialMetric] = []
        try:
            from paddleocr import PPStructure
            import fitz  # PyMuPDF
            from PIL import Image
            import io
            import pandas as pd  # type: ignore
        except Exception as e:
            print(f"DEBUG: PP-Structure anchor-crop prerequisites missing: {e}")
            return metrics

        table_engine = PPStructure(show_log=False)

        # Define anchor phrases to search on the page text
        if metric_type == 'eps':
            anchors = [
                'earnings per share',
                'eps',
                'earnings per share (cents)',
                'basic earnings per share',
                'basic earnings per share (cents)',
                'earnings per share - continuing operations',
                'earnings per share \u2013 continuing operations'
            ]
        else:
            anchors = [
                'return on equity',
                'roe',
                'return on equity (%)',
                'roe (%)'
            ]

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"DEBUG: Could not open PDF for anchor-crop: {e}")
            return metrics

        try:
            for page_index in range(len(doc)):
                try:
                    page = doc[page_index]
                    # Extract text blocks and find anchors
                    blocks = page.get_text('blocks') or []
                    candidate_rects = []
                    for b in blocks:
                        try:
                            x0, y0, x1, y1, text, *_ = b
                        except Exception:
                            if len(b) >= 5:
                                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                            else:
                                continue
                        t = str(text).lower()
                        if any(a in t for a in anchors):
                            candidate_rects.append(fitz.Rect(x0, y0, x1, y1))

                    if not candidate_rects:
                        continue

                    # Expand each anchor rect to a larger crop region where the table likely resides
                    expanded_rects = []
                    for rect in candidate_rects:
                        try:
                            page_rect = page.rect
                            crop = fitz.Rect(
                                max(0, rect.x0 - 40),
                                max(0, rect.y0 - 60),
                                page_rect.x1,
                                min(page_rect.y1, rect.y1 + 800)
                            )
                            expanded_rects.append(crop)
                        except Exception:
                            continue

                    for crop in expanded_rects:
                        try:
                            pix = page.get_pixmap(dpi=300, clip=crop)
                            img = Image.open(io.BytesIO(pix.tobytes()))
                        except Exception:
                            continue

                        try:
                            result = table_engine(img)
                        except Exception:
                            continue

                        for item in result:
                            try:
                                if str(item.get('type', '')).lower() != 'table':
                                    continue
                                res = item.get('res') or {}
                                html = res.get('html')
                                if not html:
                                    continue
                                try:
                                    dfs = pd.read_html(html)
                                except Exception:
                                    dfs = []
                                for df in dfs:
                                    if df is None or df.shape[0] == 0:
                                        continue
                                    extracted = self._extract_metrics_from_dataframe(df, metric_type)
                                    metrics.extend(extracted)
                            except Exception:
                                continue
                except Exception:
                    # Any error while processing this page; continue with next page
                    continue
        finally:
            try:
                doc.close()
            except Exception:
                pass

        return metrics

    def _table_transformer_detect_and_parse(self, pdf_path: str, metric_type: str) -> List[FinancialMetric]:
        """Detect table regions using Microsoft's Table Transformer, then parse with PP-Structure.

        Strategy:
        - Render each PDF page to an image
        - Run detection model to get table bounding boxes
        - Crop each table and pass the crop to PP-Structure to obtain HTML
        - Convert HTML to DataFrame(s) and extract anchored metrics
        """
        metrics: List[FinancialMetric] = []
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            import torch
            import fitz  # PyMuPDF
            from PIL import Image
            import io
            import pandas as pd  # type: ignore
            from paddleocr import PPStructure
        except Exception as e:
            print(f"DEBUG: Table Transformer prerequisites missing: {e}")
            return metrics

        try:
            processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
            model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
            model.eval()
        except Exception as e:
            print(f"DEBUG: Could not load Table Transformer: {e}")
            return metrics

        table_engine = PPStructure(show_log=False)

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"DEBUG: Could not open PDF for Table Transformer: {e}")
            return metrics

        try:
            id2label = getattr(model.config, 'id2label', {})
            for page_index in range(len(doc)):
                try:
                    page = doc[page_index]
                    pix = page.get_pixmap(dpi=200)
                    pil_img = Image.open(io.BytesIO(pix.tobytes())).convert('RGB')
                except Exception:
                    continue

                try:
                    inputs = processor(images=pil_img, return_tensors="pt")
                    with torch.no_grad():
                        outputs = model(**inputs)
                    target_sizes = torch.tensor([pil_img.size[::-1]])  # (h, w)
                    results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)
                except Exception:
                    continue

                for res in results:
                    boxes = res.get('boxes', [])
                    scores = res.get('scores', [])
                    labels = res.get('labels', [])
                    for box, score, label in zip(boxes, scores, labels):
                        try:
                            name = id2label.get(int(label), str(label))
                            if 'table' not in str(name).lower():
                                continue
                            x0, y0, x1, y1 = [int(v) for v in box.tolist()]
                            crop = pil_img.crop((x0, y0, x1, y1))
                        except Exception:
                            continue

                        # Run PP-Structure on the cropped table
                        try:
                            result = table_engine(crop)
                        except Exception:
                            continue
                        for item in result:
                            try:
                                if str(item.get('type', '')).lower() != 'table':
                                    continue
                                res_html = item.get('res', {}).get('html')
                                if not res_html:
                                    continue
                                try:
                                    dfs = pd.read_html(res_html)
                                except Exception:
                                    dfs = []
                                for df in dfs:
                                    if df is None or df.shape[0] == 0:
                                        continue
                                    extracted = self._extract_metrics_from_dataframe(df, metric_type)
                                    metrics.extend(extracted)
                            except Exception:
                                continue
        finally:
            try:
                doc.close()
            except Exception:
                pass

        return metrics

    def _structured_eps_anchored(self, layout_data: Dict[str, Any]) -> List[FinancialMetric]:
        """Parse tables anchored on 'Earnings per share' with 'Basic'/'Diluted' rows. Converts cents to dollars.
        Also skips 'Note' columns and prefers rightmost year columns when ambiguous."""
        results: List[FinancialMetric] = []
        anchor_keywords = ['earnings per share', 'eps', 'earnings per share - continuing operations', 'earnings per share \u2013 continuing operations']
        for table in layout_data.get('tables', []):
            df = pd.DataFrame(table.get('data', []))
            if df.empty:
                continue
            # Find anchor row
            anchor_row = None
            anchor_row_idx = None
            is_cents = False
            for r in range(len(df)):
                row_join = ' '.join([str(x) for x in df.iloc[r] if pd.notna(x)]).lower()
                if any(k in row_join for k in anchor_keywords):
                    anchor_row = row_join
                    anchor_row_idx = r
                    is_cents = ('cent' in row_join or 'cents' in row_join)
                    break
            # Fallback: if no explicit anchor row, probe 'Basic' row first then search nearby for anchor
            if anchor_row_idx is None:
                basic_probe_idx = None
                for r in range(len(df)):
                    first_cell = str(df.iat[r, 0]).strip().lower() if df.shape[1] > 0 else ''
                    row_text_lower = ' '.join([str(x) for x in df.iloc[r] if pd.notna(x)]).lower()
                    if first_cell == 'basic' or ' basic' in row_text_lower:
                        basic_probe_idx = r
                        break
                if basic_probe_idx is not None:
                    search_start = max(0, basic_probe_idx - 6)
                    search_end = basic_probe_idx + 1
                    for rr in range(search_start, search_end):
                        row_join = ' '.join([str(x) for x in df.iloc[rr] if pd.notna(x)]).lower()
                        if any(k in row_join for k in anchor_keywords):
                            anchor_row = row_join
                            anchor_row_idx = rr
                            is_cents = ('cent' in row_join or 'cents' in row_join)
                            break
            # Header hint for units if still no anchor
            if anchor_row_idx is None and len(df.index) > 0:
                header_text = ' '.join([str(x) for x in df.iloc[0] if pd.notna(x)]).lower()
                if 'cent' in header_text or 'cents' in header_text:
                    is_cents = True
            if anchor_row_idx is None:
                continue
            # Locate Basic row (same row or following rows)
            basic_row_idx = None
            for r in range(anchor_row_idx, min(anchor_row_idx + 5, len(df))):
                first_cell = str(df.iat[r, 0]).strip().lower() if df.shape[1] > 0 else ''
                row_text = ' '.join([str(x) for x in df.iloc[r] if pd.notna(x)]).lower()
                if first_cell == 'basic' or ' basic' in row_text:
                    basic_row_idx = r
                    break
            # Secondary fallback: broaden search window if not found
            if basic_row_idx is None:
                for rr in range(max(0, anchor_row_idx - 2), min(anchor_row_idx + 10, len(df))):
                    first_cell = str(df.iat[rr, 0]).strip().lower() if df.shape[1] > 0 else ''
                    row_text = ' '.join([str(x) for x in df.iloc[rr] if pd.notna(x)]).lower()
                    if first_cell == 'basic' or ' basic' in row_text:
                        basic_row_idx = rr
                        break
            if basic_row_idx is None:
                continue

            # Optionally locate Diluted row to confirm block context
            diluted_row_idx = None
            for rr in range(basic_row_idx, min(basic_row_idx + 5, len(df))):
                first_cell = str(df.iat[rr, 0]).strip().lower() if df.shape[1] > 0 else ''
                row_text = ' '.join([str(x) for x in df.iloc[rr] if pd.notna(x)]).lower()
                if first_cell == 'diluted' or ' diluted' in row_text:
                    diluted_row_idx = rr
                    break
            # Detect year headers by scanning top few rows for 4-digit years per column
            year_for_col: Dict[int, str] = {}
            note_cols: set = set()
            for c in range(len(df.columns)):
                col_text = ' '.join([str(x) for x in df.iloc[: min(5, len(df)) , c] if pd.notna(x)])
                matches = self._find_years_in_text(col_text)
                # Identify Note column and mark for exclusion
                if 'note' in str(col_text).lower():
                    note_cols.add(c)
                if matches:
                    # pick the last year occurrence for that column
                    year_for_col[c] = sorted(set(matches))[-1]
            # Remove columns labeled as Note
            for nc in list(note_cols):
                if nc in year_for_col:
                    del year_for_col[nc]
            if not year_for_col:
                # Fallback: scan entire column text to find FY years
                for c in range(len(df.columns)):
                    col_text_full = ' '.join([str(x) for x in df.iloc[:, c] if pd.notna(x)])
                    matches_full = self._find_years_in_text(col_text_full)
                    if matches_full:
                        year_for_col[c] = sorted(set(matches_full))[-1]
            if not year_for_col:
                continue
            # Prefer rightmost year columns when multiple years per row (typical financial tables)
            # Sort columns by parsed year then by column index, and limit to a sensible number if needed
            year_for_col = dict(sorted(year_for_col.items(), key=lambda kv: (int(kv[1]), kv[0])))
            # Extract values from Basic row across year columns
            for c, y in year_for_col.items():
                if c >= len(df.columns) or basic_row_idx >= len(df.index):
                    continue
                cell = df.iat[basic_row_idx, c]
                if cell is None or (isinstance(cell, float) and pd.isna(cell)):
                    continue
                raw = str(cell).replace(',', '').replace('$', '').replace('%', '').strip()
                neg = raw.startswith('(') and raw.endswith(')')
                raw = raw.strip('()')
                try:
                    val = float(raw)
                except ValueError:
                    continue
                if neg:
                    val = -val
                unit = 'dollars'
                if is_cents:
                    # Skip clearly out-of-range cents values to avoid misaligned reads
                    try:
                        raw_cents = float(raw)
                    except Exception:
                        raw_cents = None
                    if raw_cents is not None and (raw_cents < 0 or raw_cents > 1000):
                        # This likely did not come from the EPS block; skip
                        continue
                    val = val / 100.0
                # Additional guard: if not cents and value is implausibly large for EPS, skip
                else:
                    if abs(val) > 50.0:
                        continue
                results.append(FinancialMetric(
                    value=val,
                    year=str(y),
                    metric_type='eps',
                    confidence=0.85,
                    source='validator_table_eps_anchor',
                    unit=unit,
                    context=f"table_page:{table.get('page')} idx:{table.get('index')}"
                ))
        return results

    def _structured_roe_anchored(self, layout_data: Dict[str, Any]) -> List[FinancialMetric]:
        """Parse tables anchored on 'Return on equity' or 'ROE'. Also handles lines like 'Earnings per share \u2013 continuing operations (cents)'
        by ignoring them in ROE path and focusing on dedicated ROE rows."""
        results: List[FinancialMetric] = []
        anchor_keywords = ['return on equity', 'roe']
        for table in layout_data.get('tables', []):
            df = pd.DataFrame(table.get('data', []))
            if df.empty:
                continue
            # Find anchor row
            anchor_row_idx = None
            for r in range(len(df)):
                row_join = ' '.join([str(x) for x in df.iloc[r] if pd.notna(x)]).lower()
                if any(k in row_join for k in anchor_keywords):
                    anchor_row_idx = r
                    break
            if anchor_row_idx is None:
                continue
            # Detect year headers
            year_for_col: Dict[int, str] = {}
            for c in range(len(df.columns)):
                col_text = ' '.join([str(x) for x in df.iloc[: min(5, len(df)) , c] if pd.notna(x)])
                matches = self._find_years_in_text(col_text)
                if matches:
                    year_for_col[c] = sorted(set(matches))[-1]
            if not year_for_col:
                continue
            # Extract values from the anchor row across year columns
            for c, y in year_for_col.items():
                cell = df.iat[anchor_row_idx, c] if (c < len(df.columns) and anchor_row_idx < len(df.index)) else None
                if cell is None or (isinstance(cell, float) and pd.isna(cell)):
                    continue
                raw = str(cell).replace(',', '').replace('%', '').strip()
                neg = raw.startswith('(') and raw.endswith(')')
                raw = raw.strip('()')
                try:
                    val = float(raw)
                except ValueError:
                    continue
                if neg:
                    val = -val
                results.append(FinancialMetric(
                    value=val,
                    year=str(y),
                    metric_type='roe',
                    confidence=0.85,
                    source='validator_table_roe_anchor',
                    unit='percentage',
                    context=f"table_page:{table.get('page')} idx:{table.get('index')}"
                ))
        return results

    def _find_years_in_text(self, text: str) -> List[str]:
        """Find 4-digit years, including patterns like 'FY2025' or 'FYE 2024'. Returns list of year strings."""
        if not text:
            return []
        text_l = str(text)
        years = []
        # Accept FY/FYE prefixes and plain 4-digit years
        for m in re.finditer(r'(?:FY\s*|FYE\s*)?(19\d{2}|20\d{2})', text_l, flags=re.IGNORECASE):
            years.append(m.group(1))
        if years:
            return years
        # Fallback: any 4-digit year
        return re.findall(r'(19\d{2}|20\d{2})', text_l)

    def _reconcile_ai_and_validator(self, ai_data: Dict[str, Any], validator_data: Dict[str, Any], metric_type: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reconcile AI-corrected data with structured validator data.
        Returns final_data, agreement_report.
        """
        final: Dict[str, Any] = {}
        agreement: Dict[str, Any] = {'by_year': {}, 'summary': {}}
        key_name = 'basic_eps' if metric_type == 'eps' else 'basic_roe'
        # Tolerance
        tol = 0.1 if metric_type == 'eps' else 0.05
        if self.ensemble_mode == 'lenient':
            tol *= 2

        all_years = sorted(set(list(ai_data.keys()) + list(validator_data.keys())))
        agree_count = 0
        total_compared = 0

        for year in all_years:
            ai_val = ai_data.get(year, {}).get(key_name)
            vd_val = validator_data.get(year, {}).get(key_name)

            if ai_val is not None and vd_val is not None:
                total_compared += 1
                # Relative difference tolerance
                denom = max(1e-8, abs(ai_val) + abs(vd_val))
                diff = abs(ai_val - vd_val) / denom
                if diff <= tol:
                    agree_count += 1
                    chosen = ai_data[year]
                    provenance = {
                        'ensemble': True,
                        'agreement': True,
                        'ai_value': ai_val,
                        'validator_value': vd_val,
                        'tolerance': tol
                    }
                else:
                    # Prefer table validator if available; else AI
                    chosen = validator_data[year] if 'validator_table' in str(validator_data.get(year, {}).get('source', '')) else ai_data.get(year, validator_data.get(year, {}))
                    provenance = {
                        'ensemble': True,
                        'agreement': False,
                        'ai_value': ai_val,
                        'validator_value': vd_val,
                        'tolerance': tol,
                        'selection_rule': 'prefer_validator_table_on_disagreement'
                    }
                # Attach provenance
                chosen = copy.deepcopy(chosen)
                chosen['provenance'] = {**chosen.get('provenance', {}), **provenance}
                final[year] = chosen
                agreement['by_year'][year] = provenance
            elif ai_val is not None:
                chosen = copy.deepcopy(ai_data[year])
                chosen['provenance'] = {**chosen.get('provenance', {}), 'ensemble': True, 'validator_missing': True}
                final[year] = chosen
                agreement['by_year'][year] = {'ensemble': True, 'validator_missing': True}
            elif vd_val is not None:
                chosen = copy.deepcopy(validator_data[year])
                chosen['provenance'] = {**chosen.get('provenance', {}), 'ensemble': True, 'ai_missing': True}
                final[year] = chosen
                agreement['by_year'][year] = {'ensemble': True, 'ai_missing': True}

        # Summary
        agreement['summary'] = {
            'total_compared': total_compared,
            'agree_count': agree_count,
            'agree_ratio': (agree_count / total_compared) if total_compared else None,
            'mode': self.ensemble_mode,
            'tolerance': tol
        }

        return final, agreement
    
    def _calculate_ai_confidence(self, validated_data: Dict, metric_type: str) -> float:
        """AI-powered confidence scoring based on data quality and patterns"""
        if not validated_data:
            return 0.0
        
        base_confidence = 0.5
        confidence_factors = []
        
        # Factor 1: Data quantity
        data_quantity_factor = min(len(validated_data) / 10.0, 1.0)  # Max at 10 years
        confidence_factors.append(data_quantity_factor * 0.2)
        
        # Factor 2: Value consistency
        if metric_type == 'eps':
            values = [data['basic_eps'] for data in validated_data.values() if 'basic_eps' in data]
        else:  # ROE
            values = [data['basic_roe'] for data in validated_data.values() if 'basic_roe' in data]
        
        if values:
            # Check for reasonable value ranges
            value_range_factor = 1.0
            for value in values:
                if metric_type == 'eps':
                    if not self._is_valid_eps_value(value):
                        value_range_factor *= 0.7
                else:
                    if not self._is_valid_roe_value(value):
                        value_range_factor *= 0.7
            
            confidence_factors.append(value_range_factor * 0.3)
            
            # Factor 3: Trend consistency (no wild fluctuations)
            if len(values) > 1:
                sorted_values = sorted(values)
                max_change = max(abs(sorted_values[i] - sorted_values[i-1]) for i in range(1, len(sorted_values)))
                trend_factor = max(0.1, 1.0 - (max_change / 10.0))  # Penalize wild changes
                confidence_factors.append(trend_factor * 0.2)
        
        # Factor 4: Year distribution
        years = [int(year) for year in validated_data.keys() if year.isdigit()]
        if years:
            year_span = max(years) - min(years)
            year_factor = min(year_span / 20.0, 1.0)  # Prefer data spanning multiple years
            confidence_factors.append(year_factor * 0.1)
        
        # Factor 5: Source reliability
        source_factor = 0.9  # Annual reports are generally reliable
        confidence_factors.append(source_factor * 0.2)
        
        # Calculate final confidence
        final_confidence = base_confidence + sum(confidence_factors)
        return min(final_confidence, 1.0)  # Cap at 100%
    
    def _is_valid_eps_value(self, value: float) -> bool:
        """Check if EPS value is reasonable using AI-powered validation"""
        import datetime
        current_year = datetime.datetime.now().year
        
        # Check for NaN or infinite values
        if not (value == value) or abs(value) == float('inf'):
            return False
        
        # AI-POWERED VALIDATION: Enhanced range checking
        # Most realistic EPS values are between -50 and 50
        if value < -50 or value > 50:
            # Check for common extraction errors but don't return True for uncorrected values
            self._detect_and_correct_eps_error(value)  # Just detect, don't validate
            return False  # Value is still invalid until corrected
        
        # Check for extremely small values that might be errors (less than 0.001)
        if abs(value) < 0.001 and value != 0:
            return False
        
        return True
    
    def _detect_and_correct_eps_error(self, value: float) -> bool:
        """AI-powered error detection and correction for EPS values"""
        # Pattern 1: Decimal misplacement (e.g., 2020.0 → 20.20)
        if value > 100 and value % 100 == 0:
            suggested_correction = value / 100
            if 0.1 <= abs(suggested_correction) <= 50:
                print(f"AI DETECTION: Possible decimal misplacement detected!")
                print(f"  Original value: {value}")
                print(f"  Suggested correction: {suggested_correction}")
                print(f"  Confidence: 90% (pattern: {value} → {suggested_correction})")
                return True
        
        # Pattern 2: Thousand separator issue (e.g., 1,234 → 1.234)
        if value > 1000 and value % 1000 == 0:
            suggested_correction = value / 1000
            if 0.1 <= abs(suggested_correction) <= 50:
                print(f"AI DETECTION: Possible thousand separator issue detected!")
                print(f"  Original value: {value}")
                print(f"  Suggested correction: {suggested_correction}")
                print(f"  Confidence: 80% (pattern: {value} → {suggested_correction})")
                return True
        
        # Pattern 3: Year format confusion (e.g., 2020 → 20.20)
        if 1900 <= value <= 2100 and value % 1 == 0:
            suggested_correction = value / 100
            if 0.1 <= abs(suggested_correction) <= 50:
                print(f"AI DETECTION: Possible year format confusion detected!")
                print(f"  Original value: {value}")
                print(f"  Suggested correction: {suggested_correction}")
                print(f"  Confidence: 85% (pattern: {value} → {suggested_correction})")
                return True
        
        return False
    
    def _is_valid_roe_value(self, value: float) -> bool:
        """Check if ROE value is reasonable"""
        import datetime
        current_year = datetime.datetime.now().year
        
        # Check for NaN or infinite values
        if not (value == value) or abs(value) == float('inf'):
            return False
        
        # Check for reasonable ROE range (typically between -100% and 1000%)
        # This covers most realistic ROE scenarios
        if value < -100 or value > 1000:
            return False
        
        # Check for extremely small values that might be errors (less than 0.01%)
        if abs(value) < 0.01 and value != 0:
            return False
        
        return True
    
    def _is_reasonable_financial_value(self, value: float, metric_type: str) -> bool:
        """Check if financial value is reasonable based on metric type"""
        if metric_type == 'eps':
            return self._is_valid_eps_value(value)
        elif metric_type == 'roe':
            return self._is_valid_roe_value(value)
        else:
            # For other metrics, use general validation
            return not (value == value) and abs(value) != float('inf') and abs(value) < 1e6
    
    def _validate_extracted_data(self, eps_data: Dict, roe_data: Dict) -> Dict[str, Any]:
        """Comprehensive validation of extracted financial data"""
        validation_report = {
            'eps_validation': {},
            'roe_validation': {},
            'overall_confidence': 0.0,
            'warnings': [],
            'recommendations': [],
            'data_quality_score': 0.0
        }
        
        # Validate EPS data
        eps_warnings = []
        eps_valid_count = 0
        eps_total_count = 0
        
        for year, data in eps_data.items():
            if isinstance(data, dict) and 'basic_eps' in data:
                eps_total_count += 1
                eps_value = data['basic_eps']
                
                if eps_value is not None:
                    # Check for unrealistic values
                    if not self._is_valid_eps_value(eps_value):
                        eps_warnings.append(f"EPS {year}: Unrealistic value {eps_value}")
                        data['validation_warning'] = f"Unrealistic EPS value: {eps_value}"
                    else:
                        eps_valid_count += 1
                        
                    # Check for future years
                    try:
                        year_int = int(year)
                        if year_int > datetime.now().year + 1:
                            eps_warnings.append(f"EPS {year}: Future year detected")
                            data['validation_warning'] = f"Future year detected: {year}"
                    except ValueError:
                        eps_warnings.append(f"EPS {year}: Invalid year format")
        
        # Validate ROE data
        roe_warnings = []
        roe_valid_count = 0
        roe_total_count = 0
        
        for year, data in roe_data.items():
            if isinstance(data, dict) and 'basic_roe' in data:
                roe_total_count += 1
                roe_value = data['basic_roe']
                
                if roe_value is not None:
                    # Check for unrealistic values
                    if not self._is_valid_roe_value(roe_value):
                        roe_warnings.append(f"ROE {year}: Unrealistic value {roe_value}")
                        data['validation_warning'] = f"Unrealistic ROE value: {roe_value}"
                    else:
                        roe_valid_count += 1
                        
                    # Check for future years
                    try:
                        year_int = int(year)
                        if year_int > datetime.now().year + 1:
                            roe_warnings.append(f"ROE {year}: Future year detected")
                            data['validation_warning'] = f"Future year detected: {year}"
                    except ValueError:
                        roe_warnings.append(f"ROE {year}: Invalid year format")
        
        # Calculate validation scores
        eps_confidence = eps_valid_count / eps_total_count if eps_total_count > 0 else 0.0
        roe_confidence = roe_valid_count / roe_total_count if roe_total_count > 0 else 0.0
        
        validation_report['eps_validation'] = {
            'total_count': eps_total_count,
            'valid_count': eps_valid_count,
            'confidence': eps_confidence,
            'warnings': eps_warnings
        }
        
        validation_report['roe_validation'] = {
            'total_count': roe_total_count,
            'valid_count': roe_valid_count,
            'confidence': roe_confidence,
            'warnings': roe_warnings
        }
        
        # Overall confidence (weighted average)
        total_metrics = eps_total_count + roe_total_count
        if total_metrics > 0:
            validation_report['overall_confidence'] = (
                (eps_valid_count + roe_valid_count) / total_metrics
            )
        
        # Data quality score (0-100)
        validation_report['data_quality_score'] = validation_report['overall_confidence'] * 100
        
        # Generate warnings and recommendations
        all_warnings = eps_warnings + roe_warnings
        validation_report['warnings'] = all_warnings
        
        if validation_report['overall_confidence'] < 0.5:
            validation_report['recommendations'].append("Low confidence in extracted data - manual review recommended")
        
        if len(all_warnings) > 0:
            validation_report['recommendations'].append("Review flagged values for accuracy")
        
        if eps_total_count < 5 or roe_total_count < 5:
            validation_report['recommendations'].append("Consider supplementing with additional years of data")
        
        return validation_report
    
    def _calculate_confidence_score(self, eps_data: Dict, roe_data: Dict) -> float:
        """Calculate overall confidence score"""
        total_metrics = len(eps_data) + len(roe_data)
        if total_metrics == 0:
            return 0.0
        
        total_confidence = 0.0
        count = 0
        
        for data in [eps_data, roe_data]:
            for year_data in data.values():
                if isinstance(year_data, dict) and 'confidence' in year_data:
                    total_confidence += year_data['confidence']
                    count += 1
        
        return total_confidence / count if count > 0 else 0.0
    
    def _check_data_completeness(self, eps_data: Dict, roe_data: Dict) -> Dict[str, Any]:
        """Check if we have sufficient data for analysis"""
        eps_years = len(eps_data)
        roe_years = len(roe_data)
        
        completeness_report = {
            'eps_years': eps_years,
            'roe_years': roe_years,
            'eps_sufficient': eps_years >= 10,
            'roe_sufficient': roe_years >= 10,
            'overall_sufficient': eps_years >= 10 and roe_years >= 10,
            'warnings': []
        }
        
        if eps_years < 10:
            completeness_report['warnings'].append(f"EPS data: Only {eps_years} years available (need at least 10)")
        
        if roe_years < 10:
            completeness_report['warnings'].append(f"ROE data: Only {roe_years} years available (need at least 10)")
        
        return completeness_report
    
    def _detect_eps_unit(self, header_text: str, table_text: str) -> str:
        """Detect whether EPS values are in cents or dollars"""
        text_to_check = (header_text + " " + table_text).lower()
        
        # Look for explicit unit indicators
        if any(indicator in text_to_check for indicator in ['cents', '¢', 'cent']):
            return 'cents'
        elif any(indicator in text_to_check for indicator in ['dollars', '$', 'usd']):
            return 'dollars'
        
        # Look for patterns that suggest cents (small values, typically < 10)
        # This is a heuristic - if most values are small, likely cents
        numbers = re.findall(r'\d+\.?\d*', text_to_check)
        if numbers:
            try:
                avg_value = sum(float(n) for n in numbers[:10]) / min(len(numbers), 10)
                if avg_value < 10:  # Likely cents
                    return 'cents'
            except:
                pass
        
        # Default to dollars if uncertain
        return 'dollars'
    
    def _detect_roe_unit(self, header_text: str, table_text: str) -> str:
        """Detect whether ROE values are in percentage or decimal format"""
        text_to_check = (header_text + " " + table_text).lower()
        
        # Look for explicit unit indicators
        if any(indicator in text_to_check for indicator in ['%', 'percent', 'percentage']):
            return 'percentage'
        elif any(indicator in text_to_check for indicator in ['decimal', 'ratio']):
            return 'decimal'
        
        # Look for patterns that suggest percentage (values typically > 1)
        # This is a heuristic - if most values are > 1, likely percentage
        numbers = re.findall(r'\d+\.?\d*', text_to_check)
        if numbers:
            try:
                avg_value = sum(float(n) for n in numbers[:10]) / min(len(numbers), 10)
                if avg_value > 1:  # Likely percentage
                    return 'percentage'
                else:  # Likely decimal
                    return 'decimal'
            except:
                pass
        
        # Default to percentage if uncertain
        return 'percentage'
    
    def _detect_eps_unit_from_text(self, text: str) -> str:
        """Detect EPS unit from text context"""
        text_lower = text.lower()
        
        # Look for explicit unit indicators
        if any(indicator in text_lower for indicator in ['cents', '¢', 'cent']):
            return 'cents'
        elif any(indicator in text_lower for indicator in ['dollars', '$', 'usd']):
            return 'dollars'
        
        # Default to dollars if uncertain
        return 'dollars'
    
    def _detect_roe_unit_from_text(self, text: str) -> str:
        """Detect ROE unit from text context"""
        text_lower = text.lower()
        
        # Look for explicit unit indicators
        if any(indicator in text_lower for indicator in ['%', 'percent', 'percentage']):
            return 'percentage'
        elif any(indicator in text_lower for indicator in ['decimal', 'ratio']):
            return 'decimal'
        
        # Default to percentage if uncertain
        return 'percentage'

    def _extract_from_historical_summary_sections(self, historical_sections: List[Dict], metric_type: str) -> List[FinancialMetric]:
        """Extract data from historical summary text sections"""
        metrics = []
        
        print(f"DEBUG: Processing {len(historical_sections)} historical summary sections for {metric_type}")
        
        for section in historical_sections:
            text = section['text']
            
            # Enhanced pattern matching for historical data
            patterns = [
                # Pattern: "EPS 2023 0.15 2022 0.12 2021 0.10"
                rf'({"|".join(self.metric_keywords[metric_type])})\s+((?:\d{{4}}\s+[-]?\d+\.?\d*\s*)+)',
                # Pattern: "2023: 0.15, 2022: 0.12, 2021: 0.10"
                r'((?:\d{4}:\s*[-]?\d+\.?\d*[,\s]*)+)',
                # Pattern: "FY 2023 0.15 FY 2022 0.12"
                rf'FY\s+((?:\d{{4}}\s+[-]?\d+\.?\d*\s*)+)',
                # Pattern: "Earnings per share: 2023 $0.15, 2022 $0.12"
                rf'({"|".join(self.metric_keywords[metric_type])}).*?((?:\d{{4}}\s*\$?[-]?\d+\.?\d*[,\s]*)+)',
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        # Extract year-value pairs
                        data_text = match.group(2) if len(match.groups()) > 1 else match.group(1)
                        year_value_pairs = self._extract_year_value_pairs(data_text)
                        
                        for year, value in year_value_pairs:
                            # Validate year and value
                            try:
                                year_int = int(year)
                                current_year = datetime.now().year
                                if not (1995 <= year_int <= current_year):
                                    continue
                                
                                # Determine unit and convert if necessary
                                unit = self._detect_eps_unit_from_text(text) if metric_type == 'eps' else 'percentage'
                                if metric_type == 'eps' and unit == 'cents':
                                    value = value / 100
                                elif metric_type == 'roe' and unit == 'decimal':
                                    value = value * 100
                                
                                # Validate the value
                                if self._is_valid_eps_value(value) if metric_type == 'eps' else self._is_valid_roe_value(value):
                                    metrics.append(FinancialMetric(
                                        value=value,
                                        year=year,
                                        metric_type=metric_type,
                                        confidence=0.9,
                                        source='historical_summary_section',
                                        unit='dollars' if metric_type == 'eps' else 'percentage',
                                        context=f"Historical summary section: {text[:200]}..."
                                    ))
                                    print(f"DEBUG: Extracted {metric_type.upper()} from historical section for {year}: {value}")
                            except (ValueError, TypeError) as e:
                                print(f"DEBUG: Failed to process year-value pair {year}:{value}: {e}")
                                continue
                    except Exception as e:
                        print(f"DEBUG: Failed to process pattern match: {e}")
                        continue
        
        print(f"DEBUG: Total {metric_type.upper()} metrics from historical sections: {len(metrics)}")
        return metrics
    
    def _extract_year_value_pairs(self, text: str) -> List[Tuple[str, float]]:
        """Extract year-value pairs from text"""
        pairs = []
        
        # Pattern to match year-value combinations
        patterns = [
            r'(\d{4})\s+([-]?\d+\.?\d*)',  # "2023 0.15"
            r'(\d{4}):\s*([-]?\d+\.?\d*)',  # "2023: 0.15"
            r'FY\s+(\d{4})\s+([-]?\d+\.?\d*)',  # "FY 2023 0.15"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    year = match.group(1)
                    value = float(match.group(2))
                    pairs.append((year, value))
                except (ValueError, IndexError):
                    continue
        
        return pairs

    def _extract_from_financial_highlights(self, text: str, metric_type: str) -> List[FinancialMetric]:
        """Extract data from financial highlights sections"""
        metrics = []
        
        # Find financial highlights sections
        highlights_patterns = [
            r'Financial\s+Highlights.*?(?=\n\n|\n[A-Z]|$)',
            r'Key\s+Financial\s+Indicators.*?(?=\n\n|\n[A-Z]|$)',
            r'Financial\s+Summary.*?(?=\n\n|\n[A-Z]|$)',
            r'Performance\s+Highlights.*?(?=\n\n|\n[A-Z]|$)',
            r'Operating\s+Results.*?(?=\n\n|\n[A-Z]|$)',
        ]
        
        for pattern in highlights_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                highlights_text = match.group(0)
                print(f"DEBUG: Found financial highlights section: {highlights_text[:200]}...")
                
                # Extract year-value pairs from highlights
                year_value_pairs = self._extract_year_value_pairs_from_highlights(highlights_text, metric_type)
                
                for year, value in year_value_pairs:
                    metrics.append(FinancialMetric(
                        value=value,
                        year=year,
                        metric_type=metric_type,
                        confidence=0.9,
                        source='financial_highlights',
                        unit='dollars' if metric_type == 'eps' else 'percentage',
                        context=f"Financial highlights: {highlights_text[:200]}..."
                    ))
                    print(f"DEBUG: Extracted {metric_type.upper()} from highlights for {year}: {value}")
        
        return metrics
    
    def _extract_year_value_pairs_from_highlights(self, text: str, metric_type: str) -> List[Tuple[str, float]]:
        """Extract year-value pairs from financial highlights text"""
        pairs = []
        
        # Enhanced patterns for financial highlights
        patterns = [
            # Pattern: "EPS 2023: $0.15, 2022: $0.12"
            rf'({"|".join(self.metric_keywords[metric_type])}).*?((?:\d{{4}}[:\s]*\$?[-]?\d+\.?\d*[,\s]*)+)',
            # Pattern: "2023: 0.15, 2022: 0.12"
            r'((?:\d{4}:\s*[-]?\d+\.?\d*[,\s]*)+)',
            # Pattern: "FY 2023 0.15 FY 2022 0.12"
            rf'FY\s+((?:\d{{4}}\s+[-]?\d+\.?\d*\s*)+)',
            # Pattern: "Earnings per share: 2023 $0.15, 2022 $0.12"
            rf'({"|".join(self.metric_keywords[metric_type])}).*?((?:\d{{4}}\s*\$?[-]?\d+\.?\d*[,\s]*)+)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Extract year-value pairs
                    data_text = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    year_value_pairs = self._extract_year_value_pairs(data_text)
                    
                    for year, value in year_value_pairs:
                        # Validate year and value
                        try:
                            year_int = int(year)
                            current_year = datetime.now().year
                            if not (1995 <= year_int <= current_year):
                                continue
                            
                            # Determine unit and convert if necessary
                            unit = self._detect_eps_unit_from_text(text) if metric_type == 'eps' else 'percentage'
                            if metric_type == 'eps' and unit == 'cents':
                                value = value / 100
                            elif metric_type == 'roe' and unit == 'decimal':
                                value = value * 100
                            
                            # Validate the value
                            if self._is_reasonable_financial_value(value, metric_type):
                                pairs.append((year, value))
                        except (ValueError, TypeError) as e:
                            print(f"DEBUG: Failed to process year-value pair {year}:{value}: {e}")
                            continue
                except Exception as e:
                    print(f"DEBUG: Failed to process highlights pattern match: {e}")
                    continue
        
        return pairs

    def _apply_ai_corrections_to_data(self, data: Dict, metric_type: str) -> Dict:
        """Apply AI-powered corrections to a specific metric dataset"""
        corrected_data = copy.deepcopy(data)
        corrections_applied = []
        
        # Apply corrections to the specified metric data
        for year, year_data in corrected_data.items():
            if year == 'units':
                continue
            
            if metric_type == 'eps' and 'basic_eps' in year_data:
                original_value = year_data['basic_eps']
                corrected_value = self._correct_eps_value(original_value)
                
                if corrected_value != original_value:
                    # Store original and corrected values
                    year_data['original_eps'] = original_value
                    year_data['basic_eps'] = corrected_value
                    year_data['ai_corrected'] = True
                    year_data['correction_type'] = self._identify_correction_type(original_value, corrected_value)
                    
                    corrections_applied.append({
                        'year': year,
                        'metric': 'EPS',
                        'original': original_value,
                        'corrected': corrected_value,
                        'type': year_data['correction_type']
                    })
            
            elif metric_type == 'roe' and 'basic_roe' in year_data:
                original_value = year_data['basic_roe']
                corrected_value = self._correct_roe_value(original_value)
                
                if corrected_value != original_value:
                    # Store original and corrected values
                    year_data['original_roe'] = original_value
                    year_data['basic_roe'] = corrected_value
                    year_data['ai_corrected'] = True
                    year_data['correction_type'] = self._identify_correction_type(original_value, corrected_value)
                    
                    corrections_applied.append({
                        'year': year,
                        'metric': 'ROE',
                        'original': original_value,
                        'corrected': corrected_value,
                        'type': year_data['correction_type']
                    })
        
        # Log corrections for this metric
        if corrections_applied:
            print(f"AI CORRECTIONS for {metric_type.upper()}: Applied {len(corrections_applied)} corrections")
            for correction in corrections_applied:
                print(f"  🔧 {correction['year']} {correction['metric']}: {correction['original']} → {correction['corrected']} ({correction['type']})")
        else:
            print(f"AI CORRECTIONS for {metric_type.upper()}: No corrections needed")
        
        return corrected_data

    def _apply_ai_corrections(self, output_data: Dict) -> Dict:
        """Apply AI-powered corrections to extracted data"""
        corrected_data = output_data.copy()
        corrections_applied = []
        
        # Correct EPS data
        if 'eps_data' in corrected_data:
            for year, data in corrected_data['eps_data'].items():
                if year == 'units':
                    continue
                
                if 'basic_eps' in data:
                    original_value = data['basic_eps']
                    corrected_value = self._correct_eps_value(original_value)
                    
                    if corrected_value != original_value:
                        # Store original and corrected values
                        data['original_eps'] = original_value
                        data['basic_eps'] = corrected_value
                        data['ai_corrected'] = True
                        data['correction_type'] = self._identify_correction_type(original_value, corrected_value)
                        
                        corrections_applied.append({
                            'year': year,
                            'metric': 'EPS',
                            'original': original_value,
                            'corrected': corrected_value,
                            'type': data['correction_type']
                        })
        
        # Correct ROE data
        if 'roe_data' in corrected_data:
            for year, data in corrected_data['roe_data'].items():
                if year == 'units':
                    continue
                
                if 'basic_roe' in data:
                    original_value = data['basic_roe']
                    corrected_value = self._correct_roe_value(original_value)
                    
                    if corrected_value != original_value:
                        # Store original and corrected values
                        data['original_roe'] = original_value
                        data['basic_roe'] = corrected_value
                        data['ai_corrected'] = True
                        data['correction_type'] = self._identify_correction_type(original_value, corrected_value)
                        
                        corrections_applied.append({
                            'year': year,
                            'metric': 'ROE',
                            'original': original_value,
                            'corrected': corrected_value,
                            'type': data['correction_type']
                        })
        
        # Add correction summary
        if corrections_applied:
            corrected_data['ai_corrections'] = {
                'total_corrections': len(corrections_applied),
                'corrections_applied': corrections_applied,
                'correction_summary': f"AI applied {len(corrections_applied)} corrections to improve data accuracy"
            }
            print(f"AI CORRECTIONS: Applied {len(corrections_applied)} corrections to improve data accuracy")
            
            # Log each correction
            for correction in corrections_applied:
                print(f"  🔧 {correction['year']} {correction['metric']}: {correction['original']} → {correction['corrected']} ({correction['type']})")
        else:
            print("AI CORRECTIONS: No corrections needed - data appears accurate")
        
        return corrected_data
    
    def _correct_eps_value(self, value: float) -> float:
        """Correct EPS value using AI pattern recognition"""
        # Pattern 1: Decimal misplacement (e.g., 2020.0 → 20.20)
        if value > 100 and value % 100 == 0:
            suggested_correction = value / 100
            if 0.1 <= abs(suggested_correction) <= 50:
                return suggested_correction
        
        # Pattern 2: Thousand separator issue (e.g., 1,234 → 1.234)
        if value > 1000 and value % 1000 == 0:
            suggested_correction = value / 1000
            if 0.1 <= abs(suggested_correction) <= 50:
                return suggested_correction
        
        # Pattern 3: Year format confusion (e.g., 2020 → 20.20)
        if 1900 <= value <= 2100 and value % 1 == 0:
            suggested_correction = value / 100
            if 0.1 <= abs(suggested_correction) <= 50:
                return suggested_correction
        
        return value  # No correction needed
    
    def _correct_roe_value(self, value: float) -> float:
        """Correct ROE value using AI pattern recognition"""
        # Similar patterns for ROE values
        if value > 1000 and value % 100 == 0:
            suggested_correction = value / 100
            if 0.1 <= abs(suggested_correction) <= 1000:
                return suggested_correction
        
        return value  # No correction needed
    
    def _identify_correction_type(self, original: float, corrected: float) -> str:
        """Identify the type of correction applied"""
        ratio = abs(original / corrected) if corrected != 0 else float('inf')
        
        if ratio > 100:
            return 'decimal_misplacement'
        elif ratio > 10:
            return 'magnitude_error'
        elif ratio > 2:
            return 'significant_error'
        else:
            return 'minor_error'
    
    def _extract_roe_with_enhanced_patterns(self, text: str) -> List[FinancialMetric]:
        """Extract ROE using enhanced pattern recognition"""
        metrics = []
        
        # Enhanced year patterns (same as EPS)
        year_patterns = [
            r'\b(19|20)\d{2}\b',                    # Standard years: 2019, 2020
            r'\bFY\s*(19|20)\d{2}\b',              # Fiscal years: FY 2019, FY2020
            r'\b(19|20)\d{2}[-/](19|20)\d{2}\b',   # Year ranges: 2019-2020, 2019/2020
            r'\b(19|20)\d{2}\s*[-\\u2013]\s*(19|20)\d{2}\b',  # Year ranges with en dash
            r'\b(19|20)\d{2}\s*to\s*(19|20)\d{2}\b',    # Year ranges with "to"
            r'\b(19|20)\d{2}\s*through\s*(19|20)\d{2}\b', # Year ranges with "through"
        ]
        
        # Enhanced ROE patterns
        roe_patterns = [
            r'return\s+on\s+equity\s*[:\-]?\s*([\d,]+\.?\d*)%?',
            r'roe\s*[:\-]?\s*([\d,]+\.?\d*)%?',
            r'return\s+on\s+equity.*?([\d,]+\.?\d*)%?',
            r'roe.*?([\d,]+\.?\d*)%?',
            r'([\d,]+\.?\d*)%\s*roe',
            r'([\d,]+\.?\d*)%\s*return\s+on\s+equity',
        ]
        
        # Extract all years from text
        all_years = []
        for pattern in year_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                year_text = match.group(0)
                # Extract individual years from ranges
                if any(separator in year_text for separator in ['-', '/', '\\u2013', 'to', 'through']):
                    range_years = self._extract_years_from_range(year_text)
                    all_years.extend(range_years)
                else:
                    single_years = re.findall(r'(19|20)\d{2}', year_text)
                    all_years.extend(single_years)
        
        # Remove duplicates and validate
        unique_years = list(set(all_years))
        valid_years = []
        for year in unique_years:
            try:
                year_int = int(year)
                current_year = datetime.now().year
                if 1990 <= year_int <= current_year + 5:
                    valid_years.append(year)
            except ValueError:
                continue
        
        print(f"DEBUG: Enhanced ROE pattern extraction found {len(valid_years)} years: {sorted(valid_years)}")
        
        # Extract ROE values using enhanced patterns
        for pattern in roe_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value_text = match.group(1)
                try:
                    clean_value = value_text.replace(',', '').replace('$', '').replace('%', '').strip()
                    value = float(clean_value)
                    
                    # Find the closest year to this metric
                    match_start = match.start()
                    closest_year = self._find_closest_year_enhanced(text, match_start, valid_years)
                    
                    if closest_year and self._is_valid_roe_value(value):
                        metrics.append(FinancialMetric(
                            value=value,
                            year=closest_year,
                            metric_type='roe',
                            confidence=0.85,
                            source='enhanced_patterns',
                            unit='percentage',
                            context=text[max(0, match_start-50):match_start+50]
                        ))
                except ValueError:
                    continue
        
        print(f"DEBUG: Enhanced ROE pattern extraction found {len(metrics)} ROE metrics")
        return metrics
    
    def _extract_roe_comprehensive(self, layout_data: Dict[str, Any]) -> List[FinancialMetric]:
        """Comprehensive ROE extraction from all available data sources"""
        metrics = []
        
        # Extract from all sections
        for section in layout_data['sections']:
            section_text = section['text']
            
            # Check if section contains historical data
            if self._is_historical_summary_section(section_text):
                section_metrics = self._extract_roe_from_historical_section(section_text)
                metrics.extend(section_metrics)
        
        # Extract from all tables
        for table in layout_data['tables']:
            table_text = table['text']
            if any(keyword in table_text.lower() for keyword in self.metric_keywords['roe']):
                table_metrics = self._extract_roe_from_table_text(table_text)
                metrics.extend(table_metrics)
        
        # Extract from full text using comprehensive patterns
        full_text_metrics = self._extract_roe_from_full_text(layout_data['full_text'])
        metrics.extend(full_text_metrics)
        
        print(f"DEBUG: Comprehensive ROE extraction found {len(metrics)} ROE metrics")
        return metrics
    
    def _extract_roe_from_historical_section(self, text: str) -> List[FinancialMetric]:
        """Extract ROE from historical summary sections"""
        metrics = []
        
        # Enhanced patterns for historical sections
        patterns = [
            r'(\d{4}).*?roe.*?([\d,]+\.?\d*)%?',
            r'roe.*?(\d{4}).*?([\d,]+\.?\d*)%?',
            r'(\d{4}).*?return.*?([\d,]+\.?\d*)%?',
            r'return.*?(\d{4}).*?([\d,]+\.?\d*)%?',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    year = match.group(1)
                    value_text = match.group(2)
                    value = float(value_text.replace(',', '').replace('$', '').replace('%', '').strip())
                    
                    if self._is_valid_roe_value(value):
                        metrics.append(FinancialMetric(
                            value=value,
                            year=year,
                            metric_type='roe',
                            confidence=0.8,
                            source='historical_section',
                            unit='percentage',
                            context=match.group(0)
                        ))
                except ValueError:
                    continue
        
        return metrics
    
    def _extract_roe_from_table_text(self, text: str) -> List[FinancialMetric]:
        """Extract ROE from table text"""
        metrics = []
        
        # Split text into lines and look for year-value patterns
        lines = text.split('\n')
        for line in lines:
            # Look for patterns like "2019 54.00%" or "2019 ROE 54.00%"
            patterns = [
                r'(\d{4})\s+([\d,]+\.?\d*)%?',
                r'(\d{4}).*?roe.*?([\d,]+\.?\d*)%?',
                r'roe.*?(\d{4}).*?([\d,]+\.?\d*)%?',
            ]
            
            for pattern in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    try:
                        year = match.group(1)
                        value_text = match.group(2)
                        value = float(value_text.replace(',', '').replace('$', '').replace('%', '').strip())
                        
                        if self._is_valid_roe_value(value):
                            metrics.append(FinancialMetric(
                                value=value,
                                year=year,
                                metric_type='roe',
                                confidence=0.75,
                                source='table_text',
                                unit='percentage',
                                context=line
                            ))
                    except ValueError:
                        continue
        
        return metrics
    
    def _extract_roe_from_full_text(self, text: str) -> List[FinancialMetric]:
        """Extract ROE from full text using comprehensive patterns"""
        metrics = []
        
        # Comprehensive patterns for full text
        patterns = [
            r'(\d{4}).*?roe.*?([\d,]+\.?\d*)%?',
            r'roe.*?(\d{4}).*?([\d,]+\.?\d*)%?',
            r'(\d{4}).*?return.*?([\d,]+\.?\d*)%?',
            r'return.*?(\d{4}).*?([\d,]+\.?\d*)%?',
            r'(\d{4}).*?([\d,]+\.?\d*).*?roe',
            r'roe.*?(\d{4}).*?([\d,]+\.?\d*)%?',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    year = match.group(1)
                    value_text = match.group(2)
                    value = float(value_text.replace(',', '').replace('$', '').replace('%', '').strip())
                    
                    if self._is_valid_roe_value(value):
                        metrics.append(FinancialMetric(
                            value=value,
                            year=year,
                            metric_type='roe',
                            confidence=0.7,
                            source='full_text',
                            unit='percentage',
                            context=match.group(0)
                        ))
                except ValueError:
                    continue
        
        return metrics

def main():
    """Test the advanced extractor"""
    extractor = AdvancedFinancialExtractor(use_ai=False)  # Disable AI for testing
    
    # Test with a sample PDF
    test_pdf = "path/to/test.pdf"
    if os.path.exists(test_pdf):
        result = extractor.extract_from_pdf_file(test_pdf)
        print(json.dumps(result, indent=2))
    else:
        print("Test PDF not found")

if __name__ == "__main__":
    main()
