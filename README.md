# Value Analysis - AI-Powered Financial Analysis Tool

[![CI/CD Pipeline](https://github.com/your-username/Value_Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/Value_Analysis/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive financial analysis tool that uses CrewAI to extract, validate, and analyze company financial data from annual reports, generating professional Word documents with historical EPS/ROE graphs and detailed analysis.

## 🚀 Features

### Data Sources & Processing
- **Reference data priority**: JSON/CSV/Markdown files as primary source with 10-year EPS/ROE/PE data
- **Perplexity API integration**: Fetches fundamentals and PE history with citations when reference data unavailable
- **Optional PDF extraction**: AI/ML or Standard extraction from annual reports as fallback
- **Hybrid data approach**: Intelligent merging from multiple sources with validation

### Financial Analysis
- **10-year historical analysis**: Comprehensive EPS and ROE tracking with unit normalization
- **Advanced valuation metrics**: Present Value calculations using Average/Lowest PE ratios
- **WACC resolution hierarchy**: GUI override → Perplexity → Config → Default (9%)
- **Safety margin analysis**: Configurable low/high safety margins for risk assessment
- **Currency-aware formatting**: Dynamic currency display (MYR, SGD, USD, etc.)

### AI & LLM Integration  
- **Hybrid LLM architecture**: Cost-effective Ollama for analysis + reliable OpenAI for market sentiment
- **Research Analyst fixed to OpenAI**: Ensures consistent market sentiment and P/E analysis
- **Multi-agent system**: Financial, Research, and Strategic analysts with specialized roles
- **Robust JSON parsing**: Advanced cleaning and extraction from LLM outputs

### User Interface & Experience
- **Modern GUI**: Intuitive Tkinter interface with organized sections
- **Flexible configuration**: Company presets (SGX/KLSE) with manual override options  
- **Real-time logging**: Progress tracking with color-coded status messages
- **Validation overrides**: WACC, safety margins, and data source selection
- **Professional reporting**: Word documents with tables, charts, and detailed analysis

## 📊 Demo

![Value Analysis Tool GUI](docs/gui-screenshot.png)

*Screenshot of the Value Analysis Tool GUI showing the main interface with LLM configuration and analysis options.*

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key (or other LLM provider)
- Perplexity API key (required; primary data source)
- Annual report PDF files (optional, for extraction)
- (Optional) Ollama server for local LLM processing
- (Required for AI extraction) ML dependencies (transformers, torch, sentence-transformers)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Value_Analysis
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   
   Choose the appropriate requirements file based on your needs:
   
   **Full installation (recommended):**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Minimal installation (core functionality only):**
   ```bash
   pip install -r requirements-minimal.txt
   ```
   
   **Development installation (includes testing and dev tools):**
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
   
   **Note:** The full requirements.txt includes AI/ML dependencies for advanced PDF extraction. Use minimal requirements if you only need basic functionality.

   ### Requirements Files Explained:
   - **`requirements.txt`**: Complete installation with all features (AI extraction, full PDF processing, etc.)
   - **`requirements-minimal.txt`**: Core functionality only (no AI/ML dependencies)
   - **`requirements-dev.txt`**: Development tools (testing, linting, documentation)

4. **Set up environment variables (optional)**
   Create a `.env` file in the project root:
   ```
   # OpenAI API Key (only needed if using OpenAI models)
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Perplexity API Key (required for enhanced financial data)
   PERPLEXITY_API_KEY=your_perplexity_api_key_here
   PERPLEXITY_MODEL=sonar-pro
   PERPLEXITY_USE_ENHANCED=true
   ```
   
   **Note**: LLM selection is fully controlled through the GUI. You can choose between OpenAI and Ollama models for each agent without any hardcoded preferences.
   MCP_OVERWRITE_WITH_PERPLEXITY=true
   MCP_YEARS=10
   PERPLEXITY_PE_YEARS=10

   # Extraction controls
   EXTRACTION_MODE=disabled   # disabled|ai|standard
   EXTRACTION_CAN_OVERWRITE=false

   # Reference fundamentals (optional)
   REFERENCE_FUNDAMENTALS_PATH=
   OVERWRITE_WITH_REFERENCE=false

   # Valuation overrides (optional)
   DISCOUNT_RATE=
   SAFETY_MARGIN_LOW=5
   SAFETY_MARGIN_HIGH=20
   ```

## 🎯 Quick Start

### Using the Enhanced GUI (Recommended)

1. **Launch the GUI**
   ```bash
   python Value_AnalysisGUI.py
   ```

2. **Configure the analysis**:
   - Annual Report Folder: select the PDF folder (optional)
   - Company Configuration: set company details (name, ticker, exchange)
   - Perplexity Settings: API key, model, search classifier, years
   - Extraction Mode: Disabled | AI/ML | Standard
   - Overwrite Policy: allow extraction to overwrite Perplexity
   - Reference Fundamentals: JSON/CSV/Markdown (supports eps, roe, pe); optional overwrite
   - Valuation Overrides: Discount Rate (WACC), Safety Margin Low/High

3. **Click "Run Analysis"** and monitor the progress in the log window

### Data Flow and Merge Policy

- Perplexity is mandatory and primary. It is fetched first and used as the source of truth.
- Annual report extraction is optional and controlled by the GUI `Extraction Mode`:
  - `disabled` (default): no PDF extraction; report uses Perplexity only.
  - `ai`: AI/ML extractor runs on PDFs.
  - `standard`: enhanced regex/table extractor runs on PDFs.
- Merge behavior is controlled by `Extraction can overwrite Perplexity`:
  - Off (default): extractor only fills years Perplexity does not have or where its value is null.
  - On: extractor may replace overlapping years from Perplexity when it has values.
- The report shows up to the most recent 10 years and preserves units/currency in `eps_data['units']`.

#### Reference Fundamentals Upload (Optional)

You can upload your own reference EPS/ROE file to serve as a cross-check. It does not need to contain 10 full years; any subset is accepted. By default, extracted values remain primary. You may optionally overwrite extracted years with your reference for the years provided.

GUI controls (left panel):

- Reference & Validation (optional)
  - Reference file: JSON, CSV, or Markdown (.md)
  - Fields: eps, roe, pe (per year)
  - Overwrite extracted values with reference file: optional

Command-line environment variables (advanced):

```
ENABLE_EXTERNAL_VALIDATION=false
REFERENCE_FUNDAMENTALS_PATH=path\to\reference.json
OVERWRITE_WITH_REFERENCE=false
```

Supported formats:

- JSON (recommended)
  - Structure:
    - eps: mapping of year -> numeric EPS
    - roe: mapping of year -> numeric ROE (percentage number without % sign)
    - units (optional):
      - eps_unit: "dollars" or "cents"
      - currency: currency code like "SGD" or "USD"
      - roe_unit (optional): "percentage"

Example JSON:

```json
{
  "eps": {
    "2016": 0.22,
    "2017": 0.25,
    "2018": 0.23,
    "2019": 0.27
  },
  "roe": {
    "2016": 12.3,
    "2017": 14.1,
    "2018": 11.5,
    "2019": 13.2
  },
  "units": {
    "eps_unit": "dollars",
    "currency": "SGD",
    "roe_unit": "percentage"
  }
}
```

- CSV / Markdown
  - Required columns: year, eps, roe
  - Optional: pe, currency, eps_unit, roe_unit

Example CSV:

```csv
year,eps,roe
2016,0.22,12.3
2017,0.25,14.1
2018,0.23,11.5
2019,0.27,13.2
```

Behavior:

- Partial years are fine; used for cross-validation only.
- If overwrite is enabled, only overlapping years in your file will replace extracted values; all other years remain from extraction.
- EPS unit and currency handling: prefer JSON and set units to ensure clarity (dollars vs cents and currency).

### Perplexity Fundamentals

- Mandatory first source for up to 10-year EPS/ROE with citations.
- Env flags:
  - `MCP_PERPLEXITY_ENABLED=true|false`
  - `MCP_OVERWRITE_WITH_PERPLEXITY=true|false`
  - `MCP_YEARS=10`
  - `PERPLEXITY_MAX_TOKENS=4000`
- Artifacts:
  - Fundamentals: `perplexity_raw_response.json`, `mcp_perplexity_fundamentals.json`
  - PE history: `perplexity_raw_response_pe.json`, `perplexity_parsed_content_pe.txt`, `perplexity_last_error_pe.txt`


The system now supports three extraction modes via GUI:

1. **Disabled**: Skip annual report extraction; use Perplexity only.
2. **AI/ML Extraction**: Uses advanced AI/ML methods for more robust extraction.
3. **Standard Extraction**: Uses enhanced regex/table strategies.

#### AI Extraction Features

**When AI extraction is enabled:**
- **Semantic Analysis**: Uses transformers for financial text classification
- **ML Clustering**: DBSCAN clustering for pattern detection
- **Advanced Table Analysis**: Pandas-based table structure analysis
- **Context-Aware Extraction**: Analyzes surrounding text for better accuracy
- **Confidence Scoring**: Assigns confidence levels to extracted data
- **Multi-Strategy Approach**: Combines AI, ML, and traditional methods

**To enable AI-based extraction:**
- Check the "Enable AI-based Financial Data Extraction" checkbox in the GUI
- This requires additional ML dependencies (transformers, torch, sentence-transformers)
- May improve accuracy for complex report formats
- Provides confidence scores for data quality assessment

**AI Dependencies Required:**
```bash
# These are automatically installed with requirements.txt
transformers==4.45.0
torch==2.3.0
sentence-transformers==2.7.0
scikit-learn==1.5.2
pandas==2.2.1
```

#### AI Extraction Implementation

The AI extractor uses a multi-strategy approach:

1. **AI-Powered Semantic Extraction**:
   - Uses transformers for financial text classification
   - Sentence-level analysis for EPS/ROE detection
   - Context-aware number extraction

2. **ML-Based Pattern Recognition**:
   - DBSCAN clustering for pattern detection
   - Advanced table structure analysis
   - Multi-strategy extraction approach

3. **Enhanced Data Quality**:
   - Confidence scoring for extracted values
   - Multiple validation strategies
   - Comprehensive error handling

4. **Fallback Mechanisms**:
   - Always falls back to standard extraction if AI fails
   - No interruption to analysis workflow
   - Maintains compatibility

#### AI Extraction Usage

**GUI Method (Recommended):**
1. Open `Value_AnalysisGUI.py`
2. Check "Enable AI-based Financial Data Extraction" checkbox
3. Run analysis - system will use AI/ML methods
4. Monitor confidence scores in the output

**Command Line Method:**
```bash
# Set environment variable to enable AI extraction
export USE_AI_EXTRACTOR=true  # Linux/macOS
set USE_AI_EXTRACTOR=true     # Windows

# Run the analysis
python Value_Analysis.py
```

**AI Extraction Benefits:**
- **Improved Accuracy**: Better handling of complex report formats
- **Semantic Understanding**: Context-aware extraction
- **Confidence Scoring**: Quality assessment of extracted data
- **Robust Fallback**: Always falls back to standard extraction if needed

### Company Configuration

The system supports dynamic company configuration:

1. **Click "Change Company"** in the GUI
2. **Choose from presets** or enter custom details:
   - Company Name (e.g., "SATS", "OCBC")
   - Ticker Symbol (e.g., "SATS.SI", "O39.SI")
   - Industry (e.g., "Transportation", "Bank")
   - Country (e.g., "Singapore")

3. **Apply changes** - the system will automatically update all configurations

### LLM Configuration

Configure different LLMs for different agents:

1. **Financial Analyst**: Typically uses GPT-3.5-turbo for financial analysis
2. **Research Analyst**: Can use GPT-4o-mini for market research
3. **Strategic Analyst**: Can use Deepseek for strategic insights

## 📊 Output

The tool generates:

- **Word Document Report**: `companyslug_analysis_report_[timestamp].docx`
  - Executive summary and analysis
  - 10-year historical EPS and ROE data tables
  - Financial calculations (CAGR, P/E ratios, projections)
  - Market sentiment analysis
  - SWOT analysis
  - **Data Validation Results, Recommendations & Valuation** (combined)
  - Strategic recommendations

- **JSON Data Files**: Extracted and validated financial data
- **Log Files**: Detailed processing logs for debugging

## ⚙️ Configuration

### Core Files

- **`Value_Analysis.py`**: Main analysis orchestrator with validation
- **`Value_AnalysisGUI.py`**: Enhanced GUI with LLM configuration and AI extractor toggle
- **`enhanced_pdf_extractor.py`**: Multi-strategy PDF data extraction (standard mode)
- **`advanced_eps_extractor.py`**: AI/ML-based financial data extraction (advanced mode)
- **`word_document_creator_fixed.py`**: Word document generation with validation results
- **`market_data_integration.py`**: Market data API integration
- **`config_manager.py`**: Dynamic company configuration management
- **`llm_config.py`**: LLM configuration and management

### Company Configuration

The system uses template-based configuration that automatically updates:

```json
{
  "company": {
    "name": "SATS",
    "ticker": "SATS.SI",
    "industry": "Transportation",
    "country": "Singapore"
  },
  "document_settings": {
    "filename_prefix": "sats_analysis_report"
  }
}
```

### LLM Configuration

Configure different LLMs for different tasks:

```json
{
  "financial_analyst": {
    "model": "gpt-3.5-turbo",
    "provider": "openai"
  },
  "research_analyst": {
    "model": "gpt-4o-mini",
    "provider": "openai"
  },
  "strategic_analyst": {
    "model": "deepseek-r1:7b",
    "provider": "ollama"
  }
}
```

## 🔧 Advanced Usage

### PDF Extraction Strategies

The system supports two extraction modes:

#### Standard Extraction (Enhanced)
1. **Historical Section Analysis**: Identifies and extracts from historical financial sections
2. **Table Analysis**: Extracts data from financial tables
3. **Pattern Matching**: Uses intelligent patterns to find EPS/ROE data
4. **Full Text Analysis**: Comprehensive text analysis for data extraction
5. **Multi-Year Sequence Detection**: Identifies and validates year sequences

#### AI-Based Extraction (Advanced)
1. **Semantic Analysis**: Uses AI models to understand financial context
2. **Table Structure Analysis**: Advanced table parsing with pandas
3. **ML Clustering**: Identifies patterns using DBSCAN clustering
4. **Context-Aware Extraction**: Analyzes surrounding text for better accuracy
5. **Confidence Scoring**: Assigns confidence levels to extracted data
6. **Multi-Strategy Approach**: Combines AI, ML, and traditional methods
7. **Advanced Pattern Recognition**: Uses transformers for financial text classification
8. **Intelligent Data Validation**: AI-powered validation of extracted values

### Data Validation

The Overall Validation Confidence is computed primarily from EPS coverage and per‑year confidences (from Perplexity). PE ratios are not included in this confidence score. Discrepancy and advisory checks are shown in the report when a reference file is provided.

<!-- Market data provider integrations removed per policy -->

### Error Handling

Robust error handling for various scenarios:

- **Insufficient Data**: Stops analysis if less than 10 years of data found
- **LLM Failures**: Automatic fallback to alternative LLMs
- **Encoding Issues**: Unicode-safe processing for all text operations
- **API Failures**: Graceful degradation with fallback data sources
- **Extraction Failures**: Multiple fallback strategies for data extraction

## 📁 Project Structure

```
Value_Analysis/
├── Value_Analysis.py    # Main analysis orchestrator with validation
├── Value_AnalysisGUI.py            # Enhanced GUI with LLM config & AI extractor
├── enhanced_pdf_extractor.py            # Multi-strategy PDF extraction (standard)
├── advanced_eps_extractor.py            # AI/ML-based extraction (advanced)
├── word_document_creator_fixed.py       # Word document generation with validation
├── market_data_integration.py           # Market data API integration
├── config_manager.py                    # Dynamic company configuration
├── llm_config.py                        # LLM configuration management
├── company_config.json                  # Company configuration file
├── custom_llm_config.json              # LLM configuration file
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
├── *.md                                # Documentation files
└── venv/                               # Virtual environment
```

## 🐛 Troubleshooting

### Common Issues

1. **LLM Configuration Issues**:
   - Ensure API keys are correctly set in `.env` file
   - For Ollama models, ensure Ollama server is running
   - Check model names are correct (e.g., `deepseek-r1:7b` for Ollama)

2. **PDF Extraction Issues**:
   - Ensure PDF files are readable and not password-protected
   - Check that files contain financial data in text format
   - Try different annual report years if extraction fails
   - Enable AI-based extraction for complex report formats

3. **AI Extractor Issues**:
   - Ensure all AI dependencies are installed: `transformers`, `torch`, `sentence-transformers`
   - Check that PDF files contain extractable text
   - Monitor confidence scores for data quality assessment
   - If AI extraction fails, system automatically falls back to standard extraction
   - Ensure sufficient RAM for AI models (recommended: 8GB+)

4. **Market Data Issues**:
   - Verify API keys in `.env` file
   - Check internet connection for API access
   - Use "Test Ticker Config" to verify market data

5. **Encoding Issues**:
   - All Unicode issues have been resolved
   - System now uses UTF-8 encoding throughout

6. **GUI Issues**:
   - Ensure virtual environment is activated
   - Check that all dependencies are installed
   - Monitor log output for error messages

7. **Validation Issues**:
   - Check that ticker symbols are correct for external validation
   - Ensure internet connection for API access
   - Review validation results in the generated report

8. **AI Dependencies Issues**:
   - If AI extraction fails, check: `pip list | grep transformers`
   - Ensure torch is installed: `pip install torch`
   - Verify sentence-transformers: `pip install sentence-transformers`
   - Check scikit-learn version: `pip show scikit-learn`

### Debug Mode

Enable verbose output in the GUI or check log files for detailed information.

## 📈 Example Output

The generated Word document includes:

- **Company Overview**: Business context and analysis summary
- **Financial Data Tables**: 10-year historical EPS and ROE data
- **Financial Calculations**: CAGR, P/E ratios, projected EPS
- **Market Analysis**: AI-generated market sentiment analysis
- **SWOT Analysis**: Strategic strengths, weaknesses, opportunities, threats
- **Data Validation Results & Recommendations**: Confidence scores, discrepancies, and action plans
- **Strategic Recommendations**: AI-generated investment insights

## 🔄 Recent Updates

### v2.4 - Reference Data Priority & Hybrid LLM Architecture (Latest)
- **Reference data as primary source**: JSON/CSV/Markdown files now take priority over all other sources
- **Hybrid LLM approach**: Research Analyst fixed to OpenAI for reliability, others configurable (Ollama/OpenAI)
- **Enhanced GUI organization**: Reorganized layout with Reference Fundamentals at top, fallback source selection
- **Company preset updates**: Added Malaysian companies (Alliance Bank, CIMB, Genting Malaysia, Bursa Malaysia)
- **Currency-aware reporting**: Dynamic currency display based on input data (MYR, SGD, USD)
- **Improved valuation metrics**: Renamed to "Estimated Intrinsic Value", added Lowest PE calculations
- **Fixed prompt issues**: Eliminated unnecessary extraction prompts when reference data is sufficient
- **Code quality improvements**: Fixed indentation issues, enhanced error handling, removed temporary files

### v2.3 - Perplexity-first, PE enrichment, and valuation updates
- Perplexity API integration for fundamentals and PE history fetching
- Added 10-year PE history computation with Average/Lowest PE ratios
- Reference file support for PE data with intelligent merge policy
- Combined validation + valuation section with corrected metric labels
- WACC resolution hierarchy: Override → Perplexity → Config → Default (9%)
- Comprehensive valuation calculations with safety margin analysis

### v2.2 - AI Extractor Integration
- ✅ **AI-Based Financial Data Extraction**: Optional advanced extraction using machine learning and AI models
- ✅ **Enhanced GUI**: Added AI extractor toggle checkbox in the GUI
- ✅ **Multi-Strategy Approach**: Combines AI, ML, and traditional extraction methods
- ✅ **Confidence Scoring**: AI-powered confidence assessment for extracted data
- ✅ **Robust Fallback**: Automatic fallback to standard extraction if AI fails
- ✅ **Semantic Analysis**: Uses transformers for financial text classification
- ✅ **ML Clustering**: DBSCAN clustering for pattern detection
- ✅ **Advanced Table Analysis**: Pandas-based table structure analysis
- ✅ **Context-Aware Extraction**: Analyzes surrounding text for better accuracy

### v2.1 - Data Validation & Recommendations
- Added comprehensive data validation against external sources
- Implemented confidence scoring and discrepancy analysis
- Added validation results section to Word documents
- Enhanced recommendations and action plans

### v2.0 - Enhanced PDF Extraction
- Multi-strategy PDF extraction
- Improved pattern matching for various report formats
- Better handling of different currencies and units
- Enhanced validation and error handling

### v1.5 - Core Features
- CrewAI-based financial analysis
- Dynamic company configuration
- Multi-LLM support
- Professional Word document generation

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

### Quick Start for Contributors

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/Value_Analysis.git
   cd Value_Analysis
   ```
3. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
4. **Install development dependencies**
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
5. **Make your changes and test them**
6. **Submit a pull request**

For more detailed information, see [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Issues & Feature Requests

- **Bug Reports**: [Create an issue](https://github.com/your-username/Value_Analysis/issues/new?template=bug_report.md)
- **Feature Requests**: [Create an issue](https://github.com/your-username/Value_Analysis/issues/new?template=feature_request.md)
- **Questions**: [Start a discussion](https://github.com/your-username/Value_Analysis/discussions)

## ⭐ Star History

If you find this project helpful, please consider giving it a star! ⭐

## 📈 Roadmap

- [ ] Support for more LLM providers (Anthropic, Cohere)
- [ ] Web-based interface
- [ ] Real-time market data integration
- [ ] Portfolio analysis features
- [ ] Advanced visualization options
- [ ] API for programmatic access

## 🙏 Acknowledgments

- CrewAI framework for AI agent orchestration
- LiteLLM for multi-provider LLM integration
- PyMuPDF and pdfplumber for PDF processing
- Yahoo Finance and Alpha Vantage for market data
- OpenAI and other LLM providers for AI capabilities
- scikit-learn for machine learning capabilities
- Transformers and PyTorch for AI-powered text analysis
- Sentence-transformers for semantic text processing 