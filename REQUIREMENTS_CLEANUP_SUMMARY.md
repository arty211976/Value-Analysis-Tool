# Requirements Files Cleanup Summary

## Overview
The requirements files have been cleaned up and organized to provide better dependency management for the Value Analysis Tool.

## Files Created/Updated

### 1. `requirements.txt` (Updated)
- **Purpose**: Complete installation with all features
- **Includes**: AI/ML dependencies, full PDF processing, all optional features
- **Size**: ~2GB download
- **Use case**: Production use, full functionality

### 2. `requirements-minimal.txt` (New)
- **Purpose**: Core functionality only
- **Includes**: Essential dependencies only, no AI/ML packages
- **Size**: ~500MB download
- **Use case**: Lightweight installation, basic functionality

### 3. `requirements-dev.txt` (New)
- **Purpose**: Development tools and testing
- **Includes**: Testing frameworks, linting tools, documentation tools
- **Size**: Additional ~200MB
- **Use case**: Development and testing

### 4. `setup.py` (New)
- **Purpose**: Interactive setup script
- **Features**: Python version checking, virtual environment detection, guided installation
- **Use case**: User-friendly setup process

## Version Conflicts Resolved

The following version conflicts were identified and resolved:

1. **pypdf**: Updated from 3.17.4 to 5.1.0 (required by embedchain)
2. **tiktoken**: Downgraded from 0.8.0 to 0.7.0 (required by embedchain)
3. **chromadb**: Downgraded from 1.0.15 to 0.5.10 (required by embedchain)
4. **litellm**: Downgraded from 1.72.6 to 1.67.0 (for stability)

## Installation Options

### For End Users
```bash
# Full installation (recommended)
pip install -r requirements.txt

# Minimal installation
pip install -r requirements-minimal.txt
```

### For Developers
```bash
# Development installation
pip install -r requirements.txt -r requirements-dev.txt

# Interactive setup
python setup.py
```

## Benefits

1. **Flexibility**: Users can choose installation level based on needs
2. **Reduced Size**: Minimal installation saves ~1.5GB for users who don't need AI features
3. **Version Stability**: Resolved conflicts ensure consistent behavior
4. **Better Organization**: Clear categorization and documentation
5. **Developer Friendly**: Separate dev requirements and interactive setup

## Dependencies Removed

- External market data providers (Yahoo Finance, Alpha Vantage, etc.) - moved to optional
- Unused packages that were installed but not used
- Conflicting versions that caused installation issues

## Notes

- All version numbers are pinned for reproducibility
- CPU-only PyTorch is used for compatibility
- Perplexity remains the primary web data source
- Ollama support via LiteLLM (no direct Ollama Python package needed)
