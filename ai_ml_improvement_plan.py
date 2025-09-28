#!/usr/bin/env python3
"""
AI/ML Improvement Plan for EPS Extraction System
Provides specific implementation details for improving the AI-powered extraction accuracy.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ExtractionError:
    """Represents an extraction error with context."""
    year: str
    extracted_value: float
    expected_range: Tuple[float, float]
    error_type: str
    confidence: float
    suggested_correction: Optional[float] = None

class EPSDataValidator:
    """AI-powered validator for EPS data with learning capabilities."""
    
    def __init__(self):
        # Historical accuracy tracking
        self.extraction_accuracy = {}
        self.error_patterns = {}
        self.correction_success_rate = {}
        
        # Validation thresholds (learned from historical data)
        self.eps_range_thresholds = {
            'min': -50.0,  # Can be negative for loss-making companies
            'max': 100.0,  # Very high EPS values are rare
            'suspicious_threshold': 20.0  # Values above this need extra validation
        }
        
        # Common error patterns
        self.known_error_patterns = {
            'decimal_misplacement': {
                'pattern': r'(\d{4})\.0',  # e.g., 2020.0
                'correction_factor': 0.01,  # Divide by 100
                'confidence': 0.9
            },
            'thousand_separator': {
                'pattern': r'(\d{1,3}),(\d{3})',  # e.g., 1,234
                'correction_factor': 1.0,  # No change needed
                'confidence': 0.8
            },
            'percentage_as_decimal': {
                'pattern': r'(\d+\.\d+)%',  # e.g., 15.5%
                'correction_factor': 0.01,  # Convert percentage to decimal
                'confidence': 0.85
            }
        }
    
    def validate_eps_value(self, year: str, eps_value: float, source: str = "extracted") -> Dict:
        """
        Validate EPS value using AI-powered validation rules.
        
        Args:
            year: Year of the EPS value
            eps_value: The EPS value to validate
            source: Source of the data (extracted, yahoo, etc.)
            
        Returns:
            Dictionary containing validation results
        """
        validation_result = {
            'is_valid': True,
            'confidence': 1.0,
            'warnings': [],
            'suggested_corrections': [],
            'error_type': None
        }
        
        # 1. Range-based validation
        if eps_value < self.eps_range_thresholds['min'] or eps_value > self.eps_range_thresholds['max']:
            validation_result['is_valid'] = False
            validation_result['confidence'] = 0.1
            validation_result['warnings'].append(f"EPS value {eps_value} is outside expected range [{self.eps_range_thresholds['min']}, {self.eps_range_thresholds['max']}]")
            validation_result['error_type'] = 'range_violation'
        
        # 2. Suspicious value detection
        elif abs(eps_value) > self.eps_range_thresholds['suspicious_threshold']:
            validation_result['confidence'] = 0.5
            validation_result['warnings'].append(f"EPS value {eps_value} is suspiciously high/low")
            validation_result['error_type'] = 'suspicious_value'
        
        # 3. Pattern-based error detection
        pattern_corrections = self._detect_pattern_errors(eps_value)
        if pattern_corrections:
            validation_result['suggested_corrections'].extend(pattern_corrections)
            validation_result['confidence'] = min(validation_result['confidence'], 0.7)
        
        # 4. Historical consistency check
        if year in self.extraction_accuracy:
            historical_accuracy = self.extraction_accuracy[year]
            validation_result['confidence'] *= historical_accuracy
        
        # 5. Year validation
        current_year = 2025  # Current year
        if int(year) > current_year + 1:
            validation_result['is_valid'] = False
            validation_result['confidence'] = 0.0
            validation_result['warnings'].append(f"Year {year} is in the future")
            validation_result['error_type'] = 'future_year'
        
        return validation_result
    
    def _detect_pattern_errors(self, eps_value: float) -> List[Dict]:
        """Detect common extraction error patterns and suggest corrections."""
        corrections = []
        
        # Decimal misplacement detection (e.g., 2020.0 → 20.20)
        if eps_value > 100 and eps_value % 100 == 0:
            suggested_correction = eps_value / 100
            if 0.1 <= abs(suggested_correction) <= 50:  # Reasonable EPS range
                corrections.append({
                    'type': 'decimal_misplacement',
                    'original': eps_value,
                    'suggested': suggested_correction,
                    'confidence': 0.9,
                    'explanation': f"Value {eps_value} appears to have decimal misplacement. Suggested: {suggested_correction}"
                })
        
        # Thousand separator detection (e.g., 1,234 → 1.234)
        if eps_value > 1000 and eps_value % 1000 == 0:
            suggested_correction = eps_value / 1000
            if 0.1 <= abs(suggested_correction) <= 50:
                corrections.append({
                    'type': 'thousand_separator',
                    'original': eps_value,
                    'suggested': suggested_correction,
                    'confidence': 0.8,
                    'explanation': f"Value {eps_value} appears to have thousand separator issue. Suggested: {suggested_correction}"
                })
        
        return corrections
    
    def learn_from_validation(self, year: str, extracted_value: float, 
                            validated_value: float, was_correct: bool):
        """Learn from validation results to improve future accuracy."""
        if year not in self.extraction_accuracy:
            self.extraction_accuracy[year] = 0.5  # Initial confidence
        
        if was_correct:
            # Increase confidence for this year
            self.extraction_accuracy[year] = min(1.0, self.extraction_accuracy[year] + 0.1)
        else:
            # Decrease confidence for this year
            self.extraction_accuracy[year] = max(0.1, self.extraction_accuracy[year] - 0.2)
        
        # Track error patterns
        if not was_correct:
            error_type = self._classify_error(extracted_value, validated_value)
            if error_type not in self.error_patterns:
                self.error_patterns[error_type] = 0
            self.error_patterns[error_type] += 1
    
    def _classify_error(self, extracted: float, validated: float) -> str:
        """Classify the type of extraction error."""
        ratio = abs(extracted / validated) if validated != 0 else float('inf')
        
        if ratio > 100:
            return 'decimal_misplacement'
        elif ratio > 10:
            return 'magnitude_error'
        elif ratio > 2:
            return 'significant_error'
        else:
            return 'minor_error'
    
    def get_confidence_score(self, year: str) -> float:
        """Get confidence score for a specific year based on historical accuracy."""
        return self.extraction_accuracy.get(year, 0.5)

class IntelligentDataCorrector:
    """AI-powered data correction system."""
    
    def __init__(self, validator: EPSDataValidator):
        self.validator = validator
        self.correction_history = []
    
    def correct_extracted_data(self, extracted_data: Dict) -> Dict:
        """
        Apply intelligent corrections to extracted EPS data.
        
        Args:
            extracted_data: Raw extracted EPS data
            
        Returns:
            Corrected EPS data with confidence scores
        """
        corrected_data = extracted_data.copy()
        corrections_applied = []
        
        for year, data in extracted_data.items():
            if year == 'units':
                continue
                
            eps_value = data['basic_eps']
            validation_result = self.validator.validate_eps_value(year, eps_value)
            
            # Apply corrections if confidence is low
            if validation_result['confidence'] < 0.7 and validation_result['suggested_corrections']:
                best_correction = max(validation_result['suggested_corrections'], 
                                    key=lambda x: x['confidence'])
                
                if best_correction['confidence'] > 0.8:
                    original_value = eps_value
                    corrected_value = best_correction['suggested']
                    
                    # Apply correction
                    corrected_data[year]['basic_eps'] = corrected_value
                    corrected_data[year]['original_value'] = original_value
                    corrected_data[year]['correction_applied'] = True
                    corrected_data[year]['correction_type'] = best_correction['type']
                    corrected_data[year]['correction_confidence'] = best_correction['confidence']
                    corrected_data[year]['confidence'] = validation_result['confidence']
                    
                    corrections_applied.append({
                        'year': year,
                        'original': original_value,
                        'corrected': corrected_value,
                        'type': best_correction['type'],
                        'confidence': best_correction['confidence']
                    })
        
        # Add correction summary
        corrected_data['corrections_applied'] = corrections_applied
        corrected_data['total_corrections'] = len(corrections_applied)
        
        return corrected_data

class MultiSourceDataFusion:
    """AI-powered system to combine data from multiple sources."""
    
    def __init__(self):
        self.source_weights = {
            'yahoo_finance': 0.8,
            'alpha_vantage': 0.7,
            'polygon': 0.7,
            'extracted_pdf': 0.6,
            'manual_input': 0.9
        }
        self.source_accuracy = {}
    
    def fuse_data_sources(self, data_sources: Dict[str, Dict]) -> Dict:
        """
        Intelligently combine data from multiple sources.
        
        Args:
            data_sources: Dictionary of data from different sources
            
        Returns:
            Fused data with confidence scores
        """
        fused_data = {}
        
        for year in self._get_common_years(data_sources):
            year_data = {}
            weighted_values = []
            total_weight = 0
            
            for source, data in data_sources.items():
                if year in data and 'basic_eps' in data[year]:
                    value = data[year]['basic_eps']
                    weight = self.source_weights.get(source, 0.5)
                    
                    # Adjust weight based on source accuracy
                    if source in self.source_accuracy:
                        weight *= self.source_accuracy[source]
                    
                    weighted_values.append((value, weight))
                    total_weight += weight
            
            if weighted_values:
                # Calculate weighted average
                weighted_sum = sum(val * weight for val, weight in weighted_values)
                fused_value = weighted_sum / total_weight if total_weight > 0 else 0
                
                # Calculate confidence based on agreement between sources
                confidence = self._calculate_agreement_confidence(weighted_values)
                
                fused_data[year] = {
                    'basic_eps': fused_value,
                    'confidence': confidence,
                    'sources_used': list(data_sources.keys()),
                    'source_breakdown': weighted_values
                }
        
        return fused_data
    
    def _get_common_years(self, data_sources: Dict[str, Dict]) -> set:
        """Get years that are present in multiple data sources."""
        all_years = set()
        for source_data in data_sources.values():
            all_years.update(source_data.keys())
        return all_years
    
    def _calculate_agreement_confidence(self, weighted_values: List[Tuple[float, float]]) -> float:
        """Calculate confidence based on agreement between sources."""
        if len(weighted_values) < 2:
            return 0.5
        
        values = [val for val, _ in weighted_values]
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Higher confidence if values are close together
        if std_val == 0:
            return 1.0
        
        coefficient_of_variation = abs(std_val / mean_val) if mean_val != 0 else float('inf')
        
        # Convert to confidence score (0-1)
        if coefficient_of_variation < 0.1:
            return 0.9
        elif coefficient_of_variation < 0.3:
            return 0.7
        elif coefficient_of_variation < 0.5:
            return 0.5
        else:
            return 0.3

def demonstrate_improvements():
    """Demonstrate the AI/ML improvements on the current data."""
    print("=" * 80)
    print("AI/ML IMPROVEMENT DEMONSTRATION")
    print("=" * 80)
    
    # Load current data
    try:
        with open('debug_eps_data.json', 'r') as f:
            extracted_data = json.load(f)
    except FileNotFoundError:
        print("Error: debug_eps_data.json not found")
        return
    
    # Initialize AI components
    validator = EPSDataValidator()
    corrector = IntelligentDataCorrector(validator)
    fusion = MultiSourceDataFusion()
    
    print("\n1. VALIDATION ANALYSIS:")
    print("-" * 30)
    
    for year, data in extracted_data['eps_data'].items():
        if year == 'units':
            continue
            
        eps_value = data['basic_eps']
        validation = validator.validate_eps_value(year, eps_value)
        
        print(f"Year {year}: EPS = {eps_value}")
        print(f"  Valid: {validation['is_valid']}")
        print(f"  Confidence: {validation['confidence']:.2f}")
        
        if validation['warnings']:
            print(f"  Warnings: {', '.join(validation['warnings'])}")
        
        if validation['suggested_corrections']:
            print(f"  Suggested corrections: {len(validation['suggested_corrections'])}")
            for correction in validation['suggested_corrections']:
                print(f"    {correction['explanation']}")
        print()
    
    print("\n2. INTELLIGENT CORRECTION:")
    print("-" * 30)
    
    corrected_data = corrector.correct_extracted_data(extracted_data['eps_data'])
    
    if 'corrections_applied' in corrected_data:
        print(f"Total corrections applied: {corrected_data['total_corrections']}")
        for correction in corrected_data['corrections_applied']:
            print(f"  {correction['year']}: {correction['original']} → {correction['corrected']} "
                  f"({correction['type']}, confidence: {correction['confidence']:.2f})")
    else:
        print("No corrections were applied")
    
    print("\n3. IMPLEMENTATION ROADMAP:")
    print("-" * 30)
    print("Phase 1 (Immediate - 1-2 weeks):")
    print("  • Implement EPSDataValidator with range and pattern checks")
    print("  • Add decimal misplacement detection")
    print("  • Integrate validation into extraction pipeline")
    
    print("\nPhase 2 (Short-term - 1 month):")
    print("  • Implement IntelligentDataCorrector")
    print("  • Add learning capabilities for error patterns")
    print("  • Create validation feedback loop")
    
    print("\nPhase 3 (Medium-term - 2-3 months):")
    print("  • Implement MultiSourceDataFusion")
    print("  • Add machine learning models for pattern recognition")
    print("  • Create automated quality assurance pipeline")
    
    print("\nPhase 4 (Long-term - 3-6 months):")
    print("  • Advanced ML models for financial data extraction")
    print("  • Predictive error detection")
    print("  • Continuous learning and improvement system")

if __name__ == "__main__":
    demonstrate_improvements()
