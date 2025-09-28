# Financial Data Structure Standards

## Overview
This document establishes the standard naming conventions and data structures for financial data (EPS and ROE) across all files in the Value Analysis project. These standards ensure consistency, prevent naming conflicts, and facilitate future maintenance.

## Data Structure Standards

### 1. EPS (Earnings Per Share) Data Structure

#### Standard Key Names
- **Primary Key**: `basic_eps` (MANDATORY)
- **Alternative Keys**: `eps`, `value` (for backward compatibility only)
- **Metadata Keys**: `confidence`, `source`, `currency`, `unit`

#### Standard Data Format
```python
# CORRECT FORMAT - Use this structure
eps_data = {
    "2024": {
        "basic_eps": 107.0,           # MANDATORY: EPS value
        "confidence": 0.85,           # OPTIONAL: Confidence score (0.0-1.0)
        "source": "ai_extraction",    # OPTIONAL: Data source
        "currency": "SGD",            # OPTIONAL: Currency code
        "unit": "cents"               # OPTIONAL: Unit (cents/dollars)
    },
    "2023": {
        "basic_eps": 80.0,
        "confidence": 0.90,
        "source": "pdf_extraction"
    }
}

# INCORRECT FORMAT - Avoid these structures
eps_data = {
    "2024": {"eps": 107.0},          # WRONG: Use 'basic_eps'
    "2023": 80.0,                    # WRONG: Missing structure
    "2022": {"value": 112.0}         # WRONG: Use 'basic_eps'
}
```

#### Unit Standards
- **Input Units**: Accept both cents and dollars
- **Storage Units**: Store in the original unit with `unit` metadata
- **Display Units**: Convert to dollars for final output
- **Unit Values**: `"cents"` or `"dollars"`

### 2. ROE (Return on Equity) Data Structure

#### Standard Key Names
- **Primary Key**: `basic_roe` (MANDATORY)
- **Alternative Keys**: `roe`, `value` (for backward compatibility only)
- **Metadata Keys**: `confidence`, `source`, `unit`

#### Standard Data Format
```python
# CORRECT FORMAT - Use this structure
roe_data = {
    "2024": {
        "basic_roe": 9.6,            # MANDATORY: ROE value
        "confidence": 0.88,           # OPTIONAL: Confidence score (0.0-1.0)
        "source": "ai_extraction",    # OPTIONAL: Data source
        "unit": "percentage"          # OPTIONAL: Unit (percentage/decimal)
    },
    "2023": {
        "basic_roe": 7.6,
        "confidence": 0.92,
        "source": "pdf_extraction"
    }
}

# INCORRECT FORMAT - Avoid these structures
roe_data = {
    "2024": {"roe": 9.6},            # WRONG: Use 'basic_roe'
    "2023": 7.6,                     # WRONG: Missing structure
    "2022": {"value": 11.2}          # WRONG: Use 'basic_roe'
}
```

#### Unit Standards
- **Input Units**: Accept both percentage and decimal
- **Storage Units**: Store in the original unit with `unit` metadata
- **Display Units**: Always display as percentage
- **Unit Values**: `"percentage"` or `"decimal"`

## File-Specific Standards

### 1. Data Extraction Files

#### `advanced_eps_extractor.py`
- **Output Format**: Must use `basic_eps` and `basic_roe` keys
- **Validation**: Must validate data structure before returning
- **Metadata**: Must include confidence and source information

#### `enhanced_pdf_extractor.py`
- **Output Format**: Must use `basic_eps` and `basic_roe` keys
- **Legacy Support**: Can accept `eps` and `roe` keys but must convert to standard format
- **Validation**: Must validate data structure before returning

### 2. Data Processing Files

#### `Value_Analysis.py`
- **Input Handling**: Must accept both standard and legacy key formats
- **Conversion**: Must convert all data to standard format (`basic_eps`, `basic_roe`)
- **Validation**: Must validate data structure and units
- **Output**: Must maintain standard format throughout processing

### 3. Data Display Files

#### `word_document_creator_fixed.py`
- **Input Format**: Must expect standard format (`basic_eps`, `basic_roe`)
- **Fallback Handling**: Can handle legacy formats but should log warnings
- **Display Logic**: Must handle both structured and direct value formats

## Data Flow Standards

### 1. Extraction → Processing Flow
```
PDF/Report → AI/ML Extractor → Standard Format → Validation → Processing
     ↓              ↓              ↓              ↓           ↓
  Raw Data → basic_eps/basic_roe → Validation → Standardization → Analysis
```

### 2. Processing → Display Flow
```
Processed Data → Document Creation → Final Output
      ↓              ↓                ↓
  basic_eps/basic_roe → Table/Graph Creation → Word Document
```

## Validation Rules

### 1. Data Structure Validation
- **Required Keys**: `basic_eps` or `basic_roe` must be present
- **Data Types**: Values must be numeric (int/float)
- **Year Format**: Keys must be valid years (1900-current_year)
- **Metadata**: Optional but recommended for traceability

### 2. Value Validation
- **EPS Range**: -100 to 1000 (reasonable business range)
- **ROE Range**: -50 to 100 (reasonable business range)
- **Confidence**: 0.0 to 1.0 if present
- **Units**: Must be valid unit strings if present

## Migration Guidelines

### 1. For Existing Code
- **Immediate**: Update all new code to use standard format
- **Short-term**: Add conversion logic for legacy formats
- **Long-term**: Remove legacy format support

### 2. For New Code
- **Always**: Use `basic_eps` and `basic_roe` keys
- **Always**: Include metadata when possible
- **Always**: Validate data structure before processing

## Testing Standards

### 1. Unit Tests
- **Structure Tests**: Verify correct key names
- **Validation Tests**: Verify data validation logic
- **Conversion Tests**: Verify legacy format handling

### 2. Integration Tests
- **End-to-End**: Verify complete data flow
- **Format Consistency**: Verify output format consistency
- **Error Handling**: Verify graceful handling of malformed data

## Error Handling

### 1. Invalid Data Structure
```python
# Log warning and attempt conversion
if 'basic_eps' not in year_data:
    logger.warning(f"Missing 'basic_eps' key for year {year}, attempting conversion")
    # Convert legacy format to standard format
```

### 2. Missing Required Data
```python
# Skip invalid data and continue processing
if not year_data.get('basic_eps'):
    logger.warning(f"Skipping year {year}: no valid EPS data")
    continue
```

## Future Considerations

### 1. Extensibility
- **New Metrics**: Follow the same naming pattern (`basic_metric_name`)
- **Additional Metadata**: Add new optional keys as needed
- **Versioning**: Consider version field for data structure evolution

### 2. Performance
- **Caching**: Cache standardized data structures
- **Validation**: Optimize validation for large datasets
- **Conversion**: Minimize format conversion overhead

## Compliance Checklist

### For Developers
- [ ] Use `basic_eps` and `basic_roe` keys consistently
- [ ] Include metadata when possible
- [ ] Validate data structure before processing
- [ ] Handle legacy formats gracefully
- [ ] Log warnings for non-standard data

### For Code Review
- [ ] Check for consistent key naming
- [ ] Verify data structure validation
- [ ] Ensure proper error handling
- [ ] Confirm backward compatibility
- [ ] Validate unit handling

## Contact and Updates

- **Document Owner**: Development Team
- **Last Updated**: [Current Date]
- **Review Cycle**: Monthly
- **Version**: 1.0

---

**Note**: This document is a living standard. Any changes must be approved by the development team and all affected code must be updated accordingly.
