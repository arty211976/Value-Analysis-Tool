#!/usr/bin/env python3
"""
Enhanced GUI Launcher for Value Analysis Tool
Features: Better layout, real-time logging, progress tracking, company changing, LLM selection
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import subprocess
import sys
import threading
import queue
import time
import json
import logging
from pathlib import Path
from config_manager import ConfigManager
from llm_config import LLM_SETUPS, ACTIVE_CONFIG, get_llm_config

# Configure console encoding to handle Unicode characters
if sys.platform == "win32":
    import codecs
    # Set console output encoding to UTF-8
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    # Also set environment variable
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure logging to handle Unicode characters
class UnicodeSafeHandler(logging.StreamHandler):
    """Custom handler that safely handles Unicode characters"""
    def emit(self, record):
        try:
            msg = self.format(record)
            # Replace problematic Unicode characters with ASCII equivalents
            msg = msg.replace('\u26a0', '[WARNING]')
            msg = msg.replace('\u2713', '[SUCCESS]')
            msg = msg.replace('\U0001F680', '[ROCKET]')
            msg = msg.replace('\U0001F4CA', '[CHART]')
            msg = msg.replace('\U0001F50D', '[SEARCH]')
            msg = msg.replace('\U0001F3AF', '[TARGET]')
            msg = msg.replace('\U0001F527', '[TOOL]')
            msg = msg.replace('\U0001f680', '[ROCKET]')
            msg = msg.replace('\U0001f4ca', '[CHART]')
            msg = msg.replace('\U0001f50d', '[SEARCH]')
            msg = msg.replace('\U0001f3af', '[TARGET]')
            msg = msg.replace('\U0001f527', '[TOOL]')
            stream = self.stream
            stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except UnicodeEncodeError:
            # Fallback: write ASCII version
            try:
                stream.write(msg.encode('ascii', 'replace').decode('ascii'))
                stream.write(self.terminator)
                self.flush()
            except Exception:
                pass
        except Exception:
            self.handleError(record)

# Set up logging with Unicode-safe handler
logging.basicConfig(
    level=logging.INFO,
    handlers=[UnicodeSafeHandler()],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class ValueAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Value Analysis Tool - Enhanced GUI Launcher")
        
        # Set window to start maximized for better full-screen use
        self.root.state('zoomed')  # For Windows - starts maximized
        
        # Set minimum window size
        self.root.minsize(2600, 1800)
        
        # Initialize config manager
        self.config_manager = ConfigManager()
        
        # Variables
        self.folder_path = tk.StringVar()
        self.company_name = tk.StringVar()
        self.country = tk.StringVar()
        self.stock_exchange = tk.StringVar()
        self.ticker_symbol = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready to run analysis")
        
        # LLM Configuration Variables
        self.llm_config_var = tk.StringVar(value=ACTIVE_CONFIG)
        self.financial_analyst_llm = tk.StringVar()
        self.research_analyst_llm = tk.StringVar()
        self.strategic_analyst_llm = tk.StringVar()
        
        # Provider selection variables
        self.financial_analyst_provider = tk.StringVar()
        self.strategic_analyst_provider = tk.StringVar()
        
        # Store references to comboboxes for dynamic updates
        self.financial_combo = None
        self.strategic_combo = None
        
        # Advanced Extractor Configuration
        self.use_ai_extractor = tk.BooleanVar(value=False)
        # New: unified extraction mode and overwrite control
        self.extraction_mode = tk.StringVar(value="disabled")  # disabled|ai|standard
        self.extraction_can_overwrite = tk.BooleanVar(value=False)
        # Ensemble controls
        self.enable_ensemble = tk.BooleanVar(value=True)
        self.ensemble_mode = tk.StringVar(value="strict")
        # Reference data & validation controls
        self.reference_fundamentals_path = tk.StringVar()
        self.overwrite_with_reference = tk.BooleanVar(value=False)
        self.enable_external_validation = tk.BooleanVar(value=False)
        
        # Load current company info
        self.load_current_company_info()
        
        # Load current LLM configuration
        self.load_current_llm_config()
        
        # Queue for thread-safe logging
        self.log_queue = queue.Queue()
        
        # Create widgets
        self.create_widgets()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize LLM defaults after widgets are created
        self.initialize_llm_defaults()
        
    def load_current_company_info(self):
        """Load current company information from config"""
        try:
            company_info = self.config_manager.get_company_info()
            self.company_name.set(company_info.get('name', 'SATS Ltd'))
            self.country.set(company_info.get('country', 'Singapore'))
            self.stock_exchange.set('Singapore Exchange')  # Default
            self.ticker_symbol.set(company_info.get('ticker', 'S58.SI'))
            # Market provider removed; Perplexity is primary
            self.folder_path.set(r"C:\Users\Arty2\CrewAIProjects\Financial\Scripts\reports")
        except Exception as e:
            # Fallback to defaults
            self.company_name.set("SATS Ltd")
            self.country.set("Singapore")
            self.stock_exchange.set("Singapore Exchange")
            self.ticker_symbol.set("S58.SI")
            # Market provider removed; Perplexity is primary
            self.folder_path.set(r"C:\Users\Arty2\CrewAIProjects\Financial\Scripts\reports")
    
    def load_current_llm_config(self):
        """Load current LLM configuration"""
        try:
            current_config = get_llm_config(self.llm_config_var.get())
            
            # Set default LLM assignments based on current config
            self.financial_analyst_llm.set(current_config['reasoning']['model'])
            self.research_analyst_llm.set(current_config['reasoning']['model'])
            self.strategic_analyst_llm.set(current_config['reasoning']['model'])
            
        except Exception as e:
            # Fallback to default configuration
            self.financial_analyst_llm.set("deepseek-r1:7b")
            self.research_analyst_llm.set("deepseek-r1:7b")
            self.strategic_analyst_llm.set("deepseek-r1:7b")
        
    def create_widgets(self):
        """Create and arrange widgets with better full-screen layout"""
        # Main container with three columns
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid weights for responsive layout
        main_frame.columnconfigure(0, weight=1)  # Left side (inputs)
        main_frame.columnconfigure(1, weight=4)  # Middle (LLM config) - give more space to LLM config
        main_frame.columnconfigure(2, weight=2)  # Right side (logs) - give more space to logs
        main_frame.rowconfigure(0, weight=1)
        
        # Set minimum window size to ensure all content is visible
        self.root.minsize(2600, 1800)
        
        # LEFT SIDE - Input Fields
        left_frame = ttk.LabelFrame(main_frame, text="Analysis Configuration", padding="10")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Configure grid weights for left frame
        left_frame.columnconfigure(1, weight=1)  # Make the entry field expand
        left_frame.columnconfigure(2, weight=0)  # Keep browse button fixed size
        
        # Current Company Display
        current_company_frame = ttk.LabelFrame(left_frame, text="Current Company Configuration", padding="5")
        current_company_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        company_info = self.config_manager.get_company_info()
        current_company_text = f"Company: {company_info.get('name', 'Unknown')} | Ticker: {company_info.get('ticker', 'Unknown')} | Industry: {company_info.get('industry', 'Unknown')}"
        ttk.Label(current_company_frame, text=current_company_text, font=("Arial", 9, "bold")).pack(anchor="w")
        ttk.Label(current_company_frame, text="Note: Main fields below show current configuration (read-only). Use 'Change Company Config' to modify.", 
                 font=("Arial", 8), foreground="gray").pack(anchor="w")
        
        # Input fields with better spacing (read-only display)
        ttk.Label(left_frame, text="Company Name:").grid(row=1, column=0, sticky="w", pady=5)
        company_entry = ttk.Entry(left_frame, textvariable=self.company_name, width=35, state="readonly")
        company_entry.grid(row=1, column=1, sticky="w", pady=5)
        
        ttk.Label(left_frame, text="Country:").grid(row=2, column=0, sticky="w", pady=5)
        country_entry = ttk.Entry(left_frame, textvariable=self.country, width=35, state="readonly")
        country_entry.grid(row=2, column=1, sticky="w", pady=5)
        
        ttk.Label(left_frame, text="Stock Exchange:").grid(row=3, column=0, sticky="w", pady=5)
        exchange_entry = ttk.Entry(left_frame, textvariable=self.stock_exchange, width=35, state="readonly")
        exchange_entry.grid(row=3, column=1, sticky="w", pady=5)
        
        ttk.Label(left_frame, text="Ticker Symbol:").grid(row=4, column=0, sticky="w", pady=5)
        ticker_entry = ttk.Entry(left_frame, textvariable=self.ticker_symbol, width=35, state="readonly")
        ticker_entry.grid(row=4, column=1, sticky="w", pady=5)
        
        # Market Provider removed from UI
        
        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=20, padx=10)
        
        change_company_button = ttk.Button(button_frame, text="Change Company", 
                                         command=self.show_change_company_dialog, style="Accent.TButton", width=20)
        change_company_button.pack(fill=tk.X, pady=2, padx=5)
        
        run_button = ttk.Button(button_frame, text="Run Analysis", 
                               command=self.run_analysis, style="Accent.TButton", width=20)
        run_button.pack(fill=tk.X, pady=2, padx=5)
        
        clear_button = ttk.Button(button_frame, text="Clear Logs", 
                                 command=self.clear_logs, width=20)
        clear_button.pack(fill=tk.X, pady=2, padx=5)
        
        save_logs_button = ttk.Button(button_frame, text="Save Logs", 
                                     command=self.save_logs, width=20)
        save_logs_button.pack(fill=tk.X, pady=2, padx=5)
        
        # REORGANIZED ORDER: Reference Fundamentals FIRST (Primary Source)
        validation_frame = ttk.LabelFrame(left_frame, text="Reference Fundamentals (Primary Source)", padding="10")
        validation_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=10, padx=10)

        # Add explanation for primary source
        ttk.Label(validation_frame, text="📁 Primary Data Source - Upload your verified EPS/ROE data", 
                 font=("Arial", 9, "bold"), foreground="blue").pack(anchor="w")
        ttk.Label(validation_frame, text="This data will take priority over all other sources (Annual Reports, Perplexity)", 
                 font=("Arial", 8), foreground="gray").pack(anchor="w", pady=(0, 8))
        
        # Fallback data source selection (when no reference file)
        fallback_frame = ttk.LabelFrame(validation_frame, text="When no reference file provided", padding="5")
        fallback_frame.pack(fill=tk.X, pady=(8, 0))
        
        self.fallback_source = tk.StringVar(value="perplexity")
        ttk.Label(fallback_frame, text="Primary fallback source:").pack(anchor="w")
        ttk.Radiobutton(fallback_frame, text="🌐 Perplexity (web-sourced data)", 
                       variable=self.fallback_source, value="perplexity").pack(anchor="w")
        ttk.Radiobutton(fallback_frame, text="📄 Annual Reports (PDF extraction)", 
                       variable=self.fallback_source, value="extraction").pack(anchor="w")
        ttk.Radiobutton(fallback_frame, text="🔄 Both (Perplexity + PDF)", 
                       variable=self.fallback_source, value="both").pack(anchor="w")

        # Reference fundamentals upload (EPS/ROE)
        ttk.Label(validation_frame, text="Reference EPS/ROE file (JSON, CSV, or Markdown):").pack(anchor="w")
        ref_row = ttk.Frame(validation_frame)
        ref_row.pack(fill=tk.X, pady=(2, 2))
        ref_entry = ttk.Entry(ref_row, textvariable=self.reference_fundamentals_path, width=35)
        ref_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        def browse_reference_file():
            path = filedialog.askopenfilename(title="Select reference EPS/ROE file",
                                              filetypes=[("JSON/CSV/Markdown Files", "*.json;*.csv;*.md;*.markdown"), ("All Files", "*.*")])
            if path:
                self.reference_fundamentals_path.set(path)
                self.log_message(f"Reference fundamentals file set: {path}")
        ttk.Button(ref_row, text="Browse", command=browse_reference_file, width=8).pack(side=tk.LEFT, padx=(6, 0))

        # Note: No overwrite checkbox needed since Reference is now primary

        # Advanced Extractor Configuration (Second Priority)
        extractor_frame = ttk.LabelFrame(left_frame, text="Annual Report Extraction (Optional Supplement)", padding="10")
        extractor_frame.grid(row=8, column=0, columnspan=3, sticky="ew", pady=10, padx=10)
        
        ttk.Label(extractor_frame, text="📄 Supplement Reference data with PDF extraction", 
                 font=("Arial", 8), foreground="gray").pack(anchor="w", pady=(0, 4))
        
        # Reports folder selection (moved here from main config)
        folder_frame = ttk.Frame(extractor_frame)
        folder_frame.pack(fill=tk.X, pady=(4, 8))
        ttk.Label(folder_frame, text="Reports Folder:").pack(anchor="w")
        folder_row = ttk.Frame(folder_frame)
        folder_row.pack(fill=tk.X, pady=(2, 0))
        folder_entry = ttk.Entry(folder_row, textvariable=self.folder_path, width=35)
        folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        browse_button = ttk.Button(folder_row, text="Browse", command=self.browse_folder, width=8)
        browse_button.pack(side=tk.LEFT, padx=(6, 0))
        
        # Extraction mode radios
        ttk.Label(extractor_frame, text="Extraction Mode:").pack(anchor="w")
        def on_mode_change():
            self.use_ai_extractor.set(self.extraction_mode.get() == "ai")
        ttk.Radiobutton(extractor_frame, text="Disabled", variable=self.extraction_mode, value="disabled", command=on_mode_change).pack(anchor="w")
        ttk.Radiobutton(extractor_frame, text="AI/ML Extraction", variable=self.extraction_mode, value="ai", command=on_mode_change).pack(anchor="w")
        ttk.Radiobutton(extractor_frame, text="Standard Extraction (regex)", variable=self.extraction_mode, value="standard", command=on_mode_change).pack(anchor="w")
        ttk.Checkbutton(extractor_frame, text="Allow extraction to overwrite Reference data for overlapping years", variable=self.extraction_can_overwrite).pack(anchor="w", pady=(6, 0))

        # Valuation Overrides (moved to LLM Configuration section)
        self.safety_margin_low_var = tk.StringVar(value=os.getenv('SAFETY_MARGIN_LOW', '5'))
        self.safety_margin_high_var = tk.StringVar(value=os.getenv('SAFETY_MARGIN_HIGH', '20'))
        self.discount_rate_var = tk.StringVar(value=os.getenv('DISCOUNT_RATE', '9.0'))

        # Ensemble toggle
        ensemble_checkbox = ttk.Checkbutton(extractor_frame,
                                            text="Enable Ensemble (AI + Structured Validator)",
                                            variable=self.enable_ensemble)
        ensemble_checkbox.pack(anchor="w", pady=(8, 2))

        # Ensemble mode selection
        mode_row = ttk.Frame(extractor_frame)
        mode_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Label(mode_row, text="Ensemble Mode:").pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(mode_row, textvariable=self.ensemble_mode, state="readonly",
                                  values=["strict", "lenient"], width=10)
        mode_combo.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(extractor_frame, text="strict: tighter tolerance; lenient: wider tolerance",
                  font=("Arial", 8), foreground="gray").pack(anchor="w", pady=(2, 0))

        # Perplexity settings (Third Priority)
        mcp_frame = ttk.LabelFrame(left_frame, text="Perplexity Settings (Optional Supplement)", padding="10")
        mcp_frame.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(10, 0), padx=10)
        
        ttk.Label(mcp_frame, text="🌐 Supplement Reference data with web-sourced fundamentals", 
                 font=("Arial", 8), foreground="gray").pack(anchor="w", pady=(0, 4))

        self.enable_mcp_perplexity = tk.BooleanVar(value=True)
        self.overwrite_with_mcp = tk.BooleanVar(value=True)
        self.mcp_years = tk.IntVar(value=10)

        ttk.Checkbutton(mcp_frame,
                         text="Enable Perplexity fundamentals",
                         variable=self.enable_mcp_perplexity).pack(anchor="w")

        ttk.Checkbutton(mcp_frame,
                         text="Allow Perplexity to supplement missing years (fills gaps only)",
                         variable=self.overwrite_with_mcp).pack(anchor="w", pady=(4, 0))

        years_row = ttk.Frame(mcp_frame)
        years_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(years_row, text="Years to fetch:").pack(side=tk.LEFT)
        ttk.Entry(years_row, textvariable=self.mcp_years, width=6).pack(side=tk.LEFT, padx=(6, 0))
        
        # Status
        status_frame = ttk.Frame(left_frame)
        status_frame.grid(row=10, column=0, columnspan=3, sticky="ew", pady=10)
        ttk.Label(status_frame, text="Status:").pack(anchor="w")
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                font=("Arial", 9, "bold"), foreground="blue")
        status_label.pack(anchor="w")
        
        # Configuration Testing section removed (market APIs no longer used)
        
        # MIDDLE SIDE - LLM Configuration
        middle_frame = ttk.LabelFrame(main_frame, text="LLM Configuration", padding="15")
        middle_frame.grid(row=0, column=1, sticky="nsew", padx=10)
        
        # Configure middle frame to expand vertically
        middle_frame.columnconfigure(0, weight=1)
        middle_frame.rowconfigure(0, weight=1)
        
        # Agent Overview
        overview_frame = ttk.LabelFrame(middle_frame, text="Agent Overview", padding="5")
        overview_frame.pack(fill=tk.X, pady=(0, 5))
        
        overview_text = """Three specialized agents work together to create comprehensive financial analysis:

FINANCIAL: Financial Analyst: Analyzes extracted EPS/ROE data, calculates growth rates, PE ratios, and financial projections

RESEARCH: Research Analyst: Provides market sentiment analysis, industry insights, and market trend analysis

STRATEGIC: Strategic Analyst: Conducts SWOT analysis, strategic recommendations, and risk assessment

Each agent contributes specific sections to the final Word report."""
        
        overview_label = ttk.Label(overview_frame, text=overview_text, font=("Arial", 8), 
                                 justify=tk.LEFT, wraplength=280)
        overview_label.pack(anchor="w")
        
        # LLM Configuration Overview
        config_info_frame = ttk.LabelFrame(middle_frame, text="Current LLM Setup", padding="5")
        config_info_frame.pack(fill=tk.X, pady=(0, 5))
        
        current_config = get_llm_config(self.llm_config_var.get())
        config_text = f"Active Config: {self.llm_config_var.get()}\n{current_config['description']}"
        ttk.Label(config_info_frame, text=config_text, font=("Arial", 9), justify=tk.LEFT).pack(anchor="w")
        
        # LLM Configuration Selection
        config_frame = ttk.LabelFrame(middle_frame, text="LLM Setup Selection", padding="5")
        config_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(config_frame, text="LLM Configuration:").pack(anchor="w")
        config_combo = ttk.Combobox(config_frame, textvariable=self.llm_config_var, 
                                   values=list(LLM_SETUPS.keys()), state="readonly")
        config_combo.pack(fill=tk.X, pady=2)
        config_combo.bind("<<ComboboxSelected>>", self.on_llm_config_changed)

        # Agent LLM Assignment
        agent_frame = ttk.LabelFrame(middle_frame, text="Agent LLM Assignment", padding="5")
        agent_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Financial Analyst LLM
        financial_frame = ttk.Frame(agent_frame)
        financial_frame.pack(fill=tk.X, pady=(5, 2))
        
        ttk.Label(financial_frame, text="Financial Analyst:", font=("Arial", 9, "bold")).pack(anchor="w")
        
        # Provider selection for Financial Analyst
        provider_frame = ttk.Frame(financial_frame)
        provider_frame.pack(fill=tk.X, pady=(2, 0))
        
        ttk.Label(provider_frame, text="Provider:").pack(side=tk.LEFT, padx=(0, 5))
        financial_provider_combo = ttk.Combobox(provider_frame, textvariable=self.financial_analyst_provider,
                                              values=["ollama", "openai"], state="readonly", width=10)
        financial_provider_combo.pack(side=tk.LEFT, padx=(0, 10))
        financial_provider_combo.set("ollama")  # Default to ollama
        
        ttk.Label(provider_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        financial_combo = ttk.Combobox(provider_frame, textvariable=self.financial_analyst_llm, 
                                     values=self.get_models_by_provider("ollama"), state="readonly")
        financial_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Store reference for dynamic updates
        self.financial_combo = financial_combo
        
        # Bind provider change event
        financial_provider_combo.bind("<<ComboboxSelected>>", 
                                    lambda e: self.on_provider_changed(self.financial_analyst_provider, 
                                                                     self.financial_analyst_llm, 
                                                                     self.financial_combo))
        
        # Financial Analyst Description
        financial_desc = ttk.Label(agent_frame, text="FINANCIAL: Provides: EPS calculations, CAGR analysis, PE ratios, financial projections", 
                                 font=("Arial", 8), foreground="darkgreen", wraplength=250)
        financial_desc.pack(anchor="w", pady=(0, 2))
        
        # Research Analyst LLM (FIXED TO OPENAI FOR RELIABILITY)
        ttk.Label(agent_frame, text="Research Analyst LLM:", font=("Arial", 9, "bold")).pack(anchor="w", pady=(10, 2))
        research_fixed_frame = ttk.Frame(agent_frame)
        research_fixed_frame.pack(fill=tk.X, pady=2)
        ttk.Label(research_fixed_frame, text="gpt-4o-mini (OpenAI)", 
                 font=("Arial", 9), background="lightblue", relief="solid", borderwidth=1).pack(side=tk.LEFT)
        ttk.Label(research_fixed_frame, text="🔒 Fixed for reliability", 
                 font=("Arial", 8), foreground="blue").pack(side=tk.LEFT, padx=(5, 0))
        
        # Research Analyst Description
        research_desc = ttk.Label(agent_frame, text="RESEARCH: Provides: Market sentiment analysis, industry insights, market trends", 
                                font=("Arial", 8), foreground="darkblue", wraplength=250)
        research_desc.pack(anchor="w", pady=(0, 2))
        
        # Strategic Analyst LLM
        strategic_frame = ttk.Frame(agent_frame)
        strategic_frame.pack(fill=tk.X, pady=(10, 2))
        
        ttk.Label(strategic_frame, text="Strategic Analyst:", font=("Arial", 9, "bold")).pack(anchor="w")
        
        # Provider selection for Strategic Analyst
        strategic_provider_frame = ttk.Frame(strategic_frame)
        strategic_provider_frame.pack(fill=tk.X, pady=(2, 0))
        
        ttk.Label(strategic_provider_frame, text="Provider:").pack(side=tk.LEFT, padx=(0, 5))
        strategic_provider_combo = ttk.Combobox(strategic_provider_frame, textvariable=self.strategic_analyst_provider,
                                              values=["ollama", "openai"], state="readonly", width=10)
        strategic_provider_combo.pack(side=tk.LEFT, padx=(0, 10))
        strategic_provider_combo.set("ollama")  # Default to ollama
        
        ttk.Label(strategic_provider_frame, text="Model:").pack(side=tk.LEFT, padx=(0, 5))
        strategic_combo = ttk.Combobox(strategic_provider_frame, textvariable=self.strategic_analyst_llm, 
                                     values=self.get_models_by_provider("ollama"), state="readonly")
        strategic_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Store reference for dynamic updates
        self.strategic_combo = strategic_combo
        
        # Bind provider change event
        strategic_provider_combo.bind("<<ComboboxSelected>>", 
                                    lambda e: self.on_provider_changed(self.strategic_analyst_provider, 
                                                                     self.strategic_analyst_llm, 
                                                                     self.strategic_combo))
        
        # Strategic Analyst Description
        strategic_desc = ttk.Label(agent_frame, text="STRATEGIC: Provides: SWOT analysis, strategic recommendations, risk assessment", 
                                 font=("Arial", 8), foreground="darkred", wraplength=250)
        strategic_desc.pack(anchor="w", pady=(0, 2))
        
        # LLM Configuration Buttons - Ensure they're visible
        llm_button_frame = ttk.LabelFrame(middle_frame, text="LLM Configuration Actions", padding="10")
        llm_button_frame.pack(fill=tk.X, pady=(0, 10))
        
        apply_llm_button = ttk.Button(llm_button_frame, text="Apply LLM Config", 
                                     command=self.apply_llm_configuration, width=25)
        apply_llm_button.pack(fill=tk.X, pady=5, padx=10)
        
        reset_llm_button = ttk.Button(llm_button_frame, text="Reset to Default", 
                                     command=self.reset_llm_configuration, width=25)
        reset_llm_button.pack(fill=tk.X, pady=5, padx=10)
        
        # Add some bottom padding to ensure buttons are visible
        # Valuation Overrides at bottom of LLM section
        llm_valuation = ttk.LabelFrame(middle_frame, text="Valuation Overrides", padding="6")
        llm_valuation.pack(fill=tk.X, pady=(10, 10))
        # Single-row compact layout to avoid scrolling
        ttk.Label(llm_valuation, text="Safety Margin Low (%)").grid(row=0, column=0, sticky="w", padx=(0,6))
        ttk.Entry(llm_valuation, textvariable=self.safety_margin_low_var, width=6).grid(row=0, column=1, sticky="w", padx=(0,12))
        ttk.Label(llm_valuation, text="Safety Margin High (%)").grid(row=0, column=2, sticky="w", padx=(0,6))
        ttk.Entry(llm_valuation, textvariable=self.safety_margin_high_var, width=6).grid(row=0, column=3, sticky="w", padx=(0,12))
        ttk.Label(llm_valuation, text="Discount Rate (WACC, %)").grid(row=0, column=4, sticky="w", padx=(0,6))
        ttk.Entry(llm_valuation, textvariable=self.discount_rate_var, width=6).grid(row=0, column=5, sticky="w")
        
        # Add helper text for defaults
        helper_frame = ttk.Frame(llm_valuation)
        helper_frame.grid(row=1, column=0, columnspan=6, sticky="w", pady=(5, 0))
        ttk.Label(helper_frame, text="💡 Defaults: Safety Low=5%, Safety High=20%, WACC=9% (if not specified)", 
                 font=("Arial", 8), foreground="gray").pack(anchor="w")
        
        bottom_padding = ttk.Frame(middle_frame, height=20)
        bottom_padding.pack(fill=tk.X)
        
        # RIGHT SIDE - Logs
        right_frame = ttk.LabelFrame(main_frame, text="Analysis Logs", padding="10")
        right_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 0))
        
        # Log display
        self.log_text = scrolledtext.ScrolledText(right_frame, height=30, width=80, 
                                                font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Log controls (removed duplicate buttons - they're already in the left panel)
    
    def get_available_models(self):
        """Get list of available models from all configurations"""
        models = set()
        for config_name, config in LLM_SETUPS.items():
            models.add(config['tool_handling']['model'])
            models.add(config['reasoning']['model'])
        return sorted(list(models))
    
    def get_models_by_provider(self, provider):
        """Get list of available models for a specific provider"""
        if provider == "openai":
            return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        elif provider == "ollama":
            # Check what Ollama models are actually available
            try:
                import subprocess
                result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    available_models = []
                    for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                        if line.strip():
                            model_name = line.split()[0]  # First column is model name
                            available_models.append(model_name)
                    return available_models
            except:
                pass
            # Fallback to known Ollama models
            return ["deepseek-r1:7b", "deepseek-r1:8b", "mixtral:8x7b", "qwen2.5:32b", "qwen2.5:72b", "llama3.1:70b", "phi3:14b"]
        else:
            return []
    
    def update_model_dropdown(self, provider_var, model_var, combo_widget):
        """Update model dropdown based on provider selection"""
        provider = provider_var.get()
        models = self.get_models_by_provider(provider)
        
        # Update combobox values
        combo_widget['values'] = models
        
        # Set default model if current selection is not available
        current_model = model_var.get()
        if current_model not in models and models:
            # Set to first available model or a sensible default
            if provider == "ollama" and "deepseek-r1:8b" in models:
                model_var.set("deepseek-r1:8b")  # Memory-efficient model
            elif provider == "ollama" and "mixtral:8x7b" in models:
                model_var.set("mixtral:8x7b")  # Best balance model (if memory allows)
            elif provider == "openai":
                model_var.set("gpt-4o-mini")  # Good default
            else:
                model_var.set(models[0])
        elif not models:
            model_var.set("")
    
    def on_provider_changed(self, provider_var, model_var, combo_widget):
        """Handle provider selection change"""
        self.update_model_dropdown(provider_var, model_var, combo_widget)
    
    def initialize_llm_defaults(self):
        """Initialize LLM provider and model defaults"""
        try:
            # Set default providers
            self.financial_analyst_provider.set("ollama")
            self.strategic_analyst_provider.set("ollama")
            
            # Get available Ollama models
            ollama_models = self.get_models_by_provider("ollama")
            
            # Set default models based on availability (prioritize memory-efficient models)
            if "deepseek-r1:8b" in ollama_models:
                self.financial_analyst_llm.set("deepseek-r1:8b")
                self.strategic_analyst_llm.set("deepseek-r1:8b")
            elif ollama_models:
                self.financial_analyst_llm.set(ollama_models[0])
                self.strategic_analyst_llm.set(ollama_models[0])
            
            # Update dropdowns if they exist
            if self.financial_combo:
                self.financial_combo['values'] = ollama_models
            if self.strategic_combo:
                self.strategic_combo['values'] = ollama_models
                
        except Exception as e:
            self.log_message(f"Warning: Could not initialize LLM defaults: {e}")
    
    def on_llm_config_changed(self, event=None):
        """Handle LLM configuration change"""
        try:
            current_config = get_llm_config(self.llm_config_var.get())
            
            # Update agent LLM assignments based on new config
            self.financial_analyst_llm.set(current_config['reasoning']['model'])
            self.research_analyst_llm.set(current_config['reasoning']['model'])
            self.strategic_analyst_llm.set(current_config['reasoning']['model'])
            
            # Update provider variables based on new config
            self.financial_analyst_provider.set(current_config['reasoning']['provider'])
            self.strategic_analyst_provider.set(current_config['reasoning']['provider'])
            
            # Update model dropdowns based on new provider selection
            self.update_model_dropdown(self.financial_analyst_provider, self.financial_analyst_llm, self.financial_combo)
            self.update_model_dropdown(self.strategic_analyst_provider, self.strategic_analyst_llm, self.strategic_combo)
            
            self.log_message(f"LLM Configuration changed to: {self.llm_config_var.get()}")
            self.log_message(f"Description: {current_config['description']}")
            self.log_message(f"Provider updated to: {current_config['reasoning']['provider']}")
            
        except Exception as e:
            self.log_message(f"Error updating LLM configuration: {e}")
    
    def apply_llm_configuration(self):
        """Apply the current LLM configuration"""
        try:
            # Create a custom LLM configuration based on user selections
            custom_config = {
                "financial_analyst": {
                    "model": self.financial_analyst_llm.get(),
                    "provider": self.financial_analyst_provider.get()
                },
                "research_analyst": {
                    "model": "gpt-4o-mini",
                    "provider": "openai"
                },
                "strategic_analyst": {
                    "model": self.strategic_analyst_llm.get(),
                    "provider": self.strategic_analyst_provider.get()
                }
            }
            
            # Save custom configuration
            self.save_custom_llm_config(custom_config)
            
            self.log_message("LLM Configuration applied successfully!")
            self.log_message(f"Financial Analyst: {self.financial_analyst_llm.get()} ({self.financial_analyst_provider.get().title()})")
            self.log_message(f"Research Analyst: gpt-4o-mini (OpenAI - Fixed)")
            self.log_message(f"Strategic Analyst: {self.strategic_analyst_llm.get()} ({self.strategic_analyst_provider.get().title()})")
            
        except Exception as e:
            self.log_message(f"Error applying LLM configuration: {e}")
    
    def get_provider_for_model(self, model):
        """Get the provider for a given model"""
        for config_name, config in LLM_SETUPS.items():
            if config['tool_handling']['model'] == model:
                return config['tool_handling']['provider']
            if config['reasoning']['model'] == model:
                return config['reasoning']['provider']
        return "ollama"  # Default fallback
    
    def save_custom_llm_config(self, config):
        """Save custom LLM configuration to a file"""
        try:
            config_data = {
                "custom_llm_config": config,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "description": "Custom LLM configuration from GUI"
            }
            
            with open("custom_llm_config.json", "w") as f:
                json.dump(config_data, f, indent=2)
                
        except Exception as e:
            self.log_message(f"Error saving custom LLM config: {e}")
    
    def reset_llm_configuration(self):
        """Reset LLM configuration to default"""
        try:
            self.llm_config_var.set(ACTIVE_CONFIG)
            self.on_llm_config_changed()
            
            self.log_message("LLM Configuration reset to default")
            
        except Exception as e:
            self.log_message(f"Error resetting LLM configuration: {e}")
    
    def on_ai_extractor_changed(self):
        """Handle AI extractor checkbox change"""
        try:
            if self.use_ai_extractor.get():
                self.log_message("AI-based extractor enabled")
                self.log_message("Note: This requires scikit-learn and other ML dependencies")
            else:
                self.log_message("AI-based extractor disabled (using standard extraction)")
        except Exception as e:
            self.log_message(f"Error updating AI extractor setting: {e}")
    
    def setup_logging(self):
        """Setup logging system"""
        def check_log_queue():
            try:
                while True:
                    message = self.log_queue.get_nowait()
                    self.log_text.insert(tk.END, message + "\n")
                    self.log_text.see(tk.END)
                    self.log_queue.task_done()
            except queue.Empty:
                pass
            finally:
                self.root.after(100, check_log_queue)
        
        check_log_queue()
    
    def log_message(self, message):
        """Add message to log queue"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_queue.put(formatted_message)
    
    def clear_logs(self):
        """Clear the log display"""
        self.log_text.delete(1.0, tk.END)
        self.log_message("Logs cleared")
    
    def save_logs(self):
        """Save logs to file"""
        try:
            filename = f"analysis_logs_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(self.log_text.get(1.0, tk.END))
            self.log_message(f"Logs saved to {filename}")
        except Exception as e:
            self.log_message(f"Error saving logs: {e}")
    
    def browse_folder(self):
        """Browse for reports folder"""
        folder = filedialog.askdirectory(title="Select Reports Folder")
        if folder:
            self.folder_path.set(folder)
            self.log_message(f"Reports folder set to: {folder}")
    
    def validate_inputs(self):
        """Validate all inputs before running analysis"""
        errors = []
        
        if not self.folder_path.get():
            errors.append("Reports folder is required")
        elif not os.path.exists(self.folder_path.get()):
            errors.append(f"Reports folder does not exist: {self.folder_path.get()}")
        
        if not self.company_name.get():
            errors.append("Company name is required")
        
        if not self.ticker_symbol.get():
            errors.append("Ticker symbol is required")
        
        if not self.country.get():
            errors.append("Country is required")
        
        # Check if PDF files exist in the reports folder
        if self.folder_path.get() and os.path.exists(self.folder_path.get()):
            pdf_files = [f for f in os.listdir(self.folder_path.get()) 
                        if f.lower().endswith('.pdf')]
            if not pdf_files:
                errors.append("No PDF files found in reports folder")
        
        return errors
    
    def update_agent_config(self):
        """Update agent configuration with current settings"""
        try:
            # Update environment variables for the analysis
            os.environ['REPORTS_DIR'] = self.folder_path.get()
            os.environ['COMPANY_NAME'] = self.company_name.get()
            os.environ['TICKER_SYMBOL'] = self.ticker_symbol.get()
            os.environ['COUNTRY'] = self.country.get()
            os.environ['INDUSTRY'] = self.config_manager.get_company_info().get('industry', 'general')
            
            self.log_message("Agent configuration updated")
            
        except Exception as e:
            self.log_message(f"Error updating agent config: {e}")
    
    def update_pdf_extractor_config(self):
        """Update PDF extractor configuration"""
        try:
            # Set environment variables for PDF extraction
            os.environ['PDF_EXTRACTION_METHOD'] = 'enhanced'
            os.environ['EXTRACTION_CONFIDENCE_THRESHOLD'] = '0.8'
            
            self.log_message("PDF extractor configuration updated")
            
        except Exception as e:
            self.log_message(f"Error updating PDF extractor config: {e}")
    
    def update_ticker_config(self):
        """Deprecated: ticker config no longer required."""
        return
    
    def run_analysis_thread(self):
        """Run analysis in a separate thread"""
        try:
            self.log_message("Starting analysis...")
            self.log_message(f"Company: {self.company_name.get()}")
            self.log_message(f"Ticker: {self.ticker_symbol.get()}")
            self.log_message(f"Reports Folder: {self.folder_path.get()}")
            self.log_message(f"LLM Config: {self.llm_config_var.get()}")
            self.log_message(f"Financial Analyst LLM: {self.financial_analyst_llm.get()}")
            self.log_message(f"Research Analyst LLM: gpt-4o-mini (OpenAI - Fixed)")
            self.log_message(f"Strategic Analyst LLM: {self.strategic_analyst_llm.get()}")
            
            # Update configurations
            self.update_agent_config()
            self.update_pdf_extractor_config()
            # Ticker config no longer required
            
            # Set LLM configuration environment variables
            os.environ['LLM_CONFIG'] = self.llm_config_var.get()
            os.environ['FINANCIAL_ANALYST_LLM'] = self.financial_analyst_llm.get()
            os.environ['RESEARCH_ANALYST_LLM'] = 'gpt-4o-mini'  # Fixed to OpenAI
            os.environ['STRATEGIC_ANALYST_LLM'] = self.strategic_analyst_llm.get()
            
            # AI extractor configuration will be passed to subprocess
            
            # Run the analysis script with custom LLM support
            script_path = "Value_Analysis.py"
            
            if not os.path.exists(script_path):
                self.log_message(f"Error: Analysis script not found: {script_path}")
                return
            
            # Load .env into current process so keys (e.g., PERPLEXITY_API_KEY / OPENAI_API_KEY) are available
            try:
                env_path = os.path.join(os.getcwd(), '.env')
                if os.path.exists(env_path):
                    with open(env_path, 'r', encoding='utf-8') as _f:
                        for _line in _f:
                            # Handle UTF-8 BOM and common patterns
                            _line = _line.lstrip('\ufeff')
                            _s = _line.strip()
                            if not _s or _s.startswith('#') or '=' not in _s:
                                continue
                            if _s.lower().startswith('export '):
                                _s = _s[7:].strip()
                            _k, _v = _s.split('=', 1)
                            _k = _k.strip()
                            _v = _v.strip()
                            # Strip inline comments
                            _hash = _v.find('#')
                            if _hash != -1:
                                _v = _v[:_hash].strip()
                            _v = _v.strip('"').strip("'")
                            if _k and _v and _k not in os.environ:
                                os.environ[_k] = _v
            except Exception:
                pass

            # Run the analysis with proper encoding handling and environment variables
            env = os.environ.copy()
            env['USE_AI_EXTRACTOR'] = str(self.use_ai_extractor.get()).lower()
            env['EXTRACTION_MODE'] = self.extraction_mode.get()
            env['EXTRACTION_CAN_OVERWRITE'] = str(self.extraction_can_overwrite.get()).lower()
            # Pass ensemble settings to subprocess
            env['ENSEMBLE_ENABLED'] = str(self.enable_ensemble.get()).lower()
            # Force strict mode per request to reduce noisy ML/context hits
            env['ENSEMBLE_MODE'] = 'strict'
            # Safety margin and discount rate overrides
            env.setdefault('SAFETY_MARGIN_LOW', os.getenv('SAFETY_MARGIN_LOW', '5'))
            env.setdefault('SAFETY_MARGIN_HIGH', os.getenv('SAFETY_MARGIN_HIGH', '20'))
            if os.getenv('DISCOUNT_RATE'):
                env['DISCOUNT_RATE'] = os.getenv('DISCOUNT_RATE')
            # Pass validation and reference settings
            # External validation removed
            env['REFERENCE_FUNDAMENTALS_PATH'] = self.reference_fundamentals_path.get()
            env['OVERWRITE_WITH_REFERENCE'] = str(self.overwrite_with_reference.get()).lower()
            env['FALLBACK_DATA_SOURCE'] = self.fallback_source.get()

            # MCP env variables
            env['MCP_PERPLEXITY_ENABLED'] = str(self.enable_mcp_perplexity.get()).lower()
            env['MCP_OVERWRITE_WITH_PERPLEXITY'] = str(self.overwrite_with_mcp.get()).lower()
            env['MCP_YEARS'] = str(self.mcp_years.get())

            # Ensure SONAR defaults propagate to subprocess if .env not read there
            env.setdefault('PERPLEXITY_MODEL', os.getenv('PERPLEXITY_MODEL', 'sonar-pro'))
            env.setdefault('PERPLEXITY_ENABLE_SEARCH_CLASSIFIER', os.getenv('PERPLEXITY_ENABLE_SEARCH_CLASSIFIER', 'true'))
            env.setdefault('PERPLEXITY_MAX_TOKENS', os.getenv('PERPLEXITY_MAX_TOKENS', '4000'))
            # Propagate API key if present
            if os.getenv('PERPLEXITY_API_KEY'):
                env['PERPLEXITY_API_KEY'] = os.getenv('PERPLEXITY_API_KEY')

            # Valuation overrides from UI
            if self.safety_margin_low_var.get():
                env['SAFETY_MARGIN_LOW'] = self.safety_margin_low_var.get()
            if self.safety_margin_high_var.get():
                env['SAFETY_MARGIN_HIGH'] = self.safety_margin_high_var.get()
            if self.discount_rate_var.get() and self.discount_rate_var.get() != '9.0':
                env['DISCOUNT_RATE'] = self.discount_rate_var.get()
            # Note: If discount_rate is 9.0 (default), don't set env var so backend uses its default logic
            
            process = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=False,  # Don't use text mode to avoid encoding issues
                env=env  # Pass environment variables to subprocess
            )
            
            # Stream output in real-time with proper encoding handling
            error_detected = False
            for line in iter(process.stdout.readline, b''):
                try:
                    # Try UTF-8 first
                    decoded_line = line.decode('utf-8', errors='replace').strip()
                except UnicodeDecodeError:
                    try:
                        # Fallback to cp1252 (Windows default)
                        decoded_line = line.decode('cp1252', errors='replace').strip()
                    except UnicodeDecodeError:
                        # Final fallback - replace problematic characters
                        decoded_line = line.decode('ascii', errors='replace').strip()
                
                if decoded_line:
                    self.log_queue.put(decoded_line)
                    
                    # Check for specific error patterns
                    if any(error_pattern in decoded_line.lower() for error_pattern in [
                        'apiconnectionerror', 'connection error', 'ollama', 'timeout', 
                        'rate limit', 'authentication', 'api key'
                    ]):
                        error_detected = True
            
            process.wait()
            
            if process.returncode == 0:
                self.log_message("Analysis completed successfully!")
                self.status_var.set("Analysis completed successfully")
            elif process.returncode == 2:
                # Perplexity-only mode insufficient data; prompt user to enable extraction
                self.log_message("Perplexity returned insufficient fundamentals (needs 10-year EPS/ROE).")
                self.status_var.set("Perplexity insufficient; select extraction mode")
                def _prompt_enable_extraction():
                    resp = messagebox.askyesno(
                        "Enable Extraction",
                        "Perplexity did not return 10-year EPS. Enable Annual Report Extraction (AI/ML) and retry now?"
                    )
                    if resp:
                        try:
                            self.extraction_mode.set("ai")
                            self.log_message("Enabling AI/ML extraction and retrying...")
                            t = threading.Thread(target=self.run_analysis_thread)
                            t.daemon = True
                            t.start()
                        except Exception as e:
                            self.log_message(f"Failed to restart analysis: {e}")
                # Show prompt on the main thread
                self.root.after(0, _prompt_enable_extraction)
            else:
                self.log_message(f"Analysis failed with return code: {process.returncode}")
                self.status_var.set("Analysis failed")
                
                # Provide specific error guidance based on detected issues
                if error_detected:
                    self.log_message("")
                    self.log_message("[INFO] LLM CONNECTION ISSUE DETECTED")
                    self.log_message("The analysis failed due to LLM connection problems.")
                    self.log_message("")
                    self.log_message("Immediate solutions:")
                    self.log_message("1. If using Ollama models: Start Ollama server")
                    self.log_message("2. If using OpenAI: Check API key and internet connection")
                    self.log_message("3. Try switching to a different LLM configuration")
                    self.log_message("4. Check firewall/network settings")
                    self.log_message("")
                    self.log_message("To switch LLM configuration:")
                    self.log_message("1. Go to 'LLM Configuration' section")
                    self.log_message("2. Select 'Unified OpenAI' or 'Unified Ollama'")
                    self.log_message("3. Click 'Apply LLM Config'")
                    self.log_message("4. Try running analysis again")
                elif process.returncode == 1:
                    self.log_message("Common causes of analysis failure:")
                    self.log_message("1. LLM connection issues (Ollama not running, API keys invalid)")
                    self.log_message("2. Missing or invalid PDF files in reports directory")
                    self.log_message("3. Insufficient EPS/ROE data extracted from PDFs")
                    self.log_message("4. Missing required dependencies (python-docx, etc.)")
                    self.log_message("5. Network connectivity issues")
                    self.log_message("")
                    self.log_message("Troubleshooting steps:")
                    self.log_message("1. Check if Ollama server is running (if using Ollama models)")
                    self.log_message("2. Verify OpenAI API key is valid (if using OpenAI models)")
                    self.log_message("3. Ensure PDF files contain annual report data")
                    self.log_message("4. Check internet connection")
                    self.log_message("5. Try using different LLM configuration")
                
        except Exception as e:
            self.log_message(f"Error running analysis: {e}")
            self.status_var.set("Analysis error")
    
    def test_ticker_config(self):
        """Deprecated: market API testing removed."""
        self.log_message("Ticker config test is no longer required")
    
    def run_analysis(self):
        """Start the analysis process"""
        # Validate inputs
        errors = self.validate_inputs()
        if errors:
            error_message = "Please fix the following errors:\n" + "\n".join(f"- {error}" for error in errors)
            messagebox.showerror("Validation Errors", error_message)
            return
        
        # Update status
        self.status_var.set("Running analysis...")
        
        # Start analysis in separate thread
        analysis_thread = threading.Thread(target=self.run_analysis_thread)
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def clear_fields(self):
        """Clear all input fields"""
        self.folder_path.set("")
        self.company_name.set("")
        self.country.set("")
        self.stock_exchange.set("")
        self.ticker_symbol.set("")
        self.market_provider.set("")
    
    def show_change_company_dialog(self):
        """Show dialog to change company configuration"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Change Company Configuration")
        dialog.geometry("600x700")  # Increased size to fit all content
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (600 // 2)
        y = (dialog.winfo_screenheight() // 2) - (700 // 2)
        dialog.geometry(f"600x700+{x}+{y}")
        
        # Make dialog resizable
        dialog.resizable(True, True)
        dialog.minsize(600, 700)
        
        # Variables for the dialog
        new_company_name = tk.StringVar(value=self.company_name.get())
        new_ticker = tk.StringVar(value=self.ticker_symbol.get())
        new_industry = tk.StringVar(value=self.config_manager.get_company_info().get('industry', 'general'))
        new_country = tk.StringVar(value=self.country.get())
        new_exchange = tk.StringVar(value=self.config_manager.get_company_info().get('exchange', 'SGX'))
        
        # Create scrollable dialog content
        canvas = tk.Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Main content frame with padding
        main_frame = ttk.Frame(scrollable_frame, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Change Company Configuration", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Company presets
        presets_frame = ttk.LabelFrame(main_frame, text="Quick Presets", padding="10")
        presets_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Preset buttons
        preset_buttons_frame = ttk.Frame(presets_frame)
        preset_buttons_frame.pack(fill=tk.X)
        
        presets = [
            ("OCBC Bank", "O39.SI", "Bank", "Singapore", "SGX"),
            ("DBS Group", "D05.SI", "Bank", "Singapore", "SGX"),
            ("UOB Bank", "U11.SI", "Bank", "Singapore", "SGX"),
            ("SATS Ltd", "S58.SI", "Transportation", "Singapore", "SGX"),
            ("Wilmar International", "F34.SI", "Consumer Goods", "Singapore", "SGX"),
            ("Singtel", "Z74.SI", "Telecommunications", "Singapore", "SGX"),
            ("Bursa Malaysia", "1818.KL", "Financial Services", "Malaysia", "KLSE"),
            ("CIMB Group", "1023.KL", "Bank", "Malaysia", "KLSE"),
            ("Genting Malaysia", "4715.KL", "Consumer Goods", "Malaysia", "KLSE"),
            ("Alliance Bank", "2488.KL", "Bank", "Malaysia", "KLSE")
        ]
        
        # Create preset buttons in a grid
        for i, (name, ticker, industry, country, exchange) in enumerate(presets):
            row = i // 2
            col = i % 2
            
            btn = ttk.Button(preset_buttons_frame, 
                           text=f"{name}\n({ticker})",
                           command=lambda n=name, t=ticker, ind=industry, c=country, e=exchange: 
                           self.apply_preset(dialog, n, t, ind, c, e))
            btn.grid(row=row, column=col, padx=5, pady=2, sticky="ew")
        
        # Configure grid weights
        preset_buttons_frame.columnconfigure(0, weight=1)
        preset_buttons_frame.columnconfigure(1, weight=1)
        
        # Manual input fields
        manual_frame = ttk.LabelFrame(main_frame, text="Manual Configuration", padding="10")
        manual_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Company name
        ttk.Label(manual_frame, text="Company Name:").pack(anchor="w")
        company_entry = ttk.Entry(manual_frame, textvariable=new_company_name, width=50)
        company_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Ticker symbol
        ttk.Label(manual_frame, text="Ticker Symbol:").pack(anchor="w")
        ticker_entry = ttk.Entry(manual_frame, textvariable=new_ticker, width=50)
        ticker_entry.pack(fill=tk.X, pady=(0, 10))
        
        # Industry
        ttk.Label(manual_frame, text="Industry:").pack(anchor="w")
        industry_combo = ttk.Combobox(manual_frame, textvariable=new_industry, 
                                    values=["Bank", "Technology", "Healthcare", "Consumer Goods", 
                                           "Energy", "Transportation", "Manufacturing", "Real Estate", 
                                           "Telecommunications", "Utilities", "Retail", "Insurance", "general"])
        industry_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Country
        ttk.Label(manual_frame, text="Country:").pack(anchor="w")
        country_combo = ttk.Combobox(manual_frame, textvariable=new_country, 
                                   values=["Singapore", "Malaysia", "United States", "United Kingdom", "Japan", 
                                          "China", "Australia", "Canada", "Germany", "France", "Other"])
        country_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Stock Exchange
        ttk.Label(manual_frame, text="Stock Exchange:").pack(anchor="w")
        exchange_combo = ttk.Combobox(manual_frame, textvariable=new_exchange, 
                                    values=["SGX", "KLSE", "NYSE", "NASDAQ", "LSE", "TSE", "HKEX", "ASX", "TSX"])
        exchange_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Auto-detect exchange from ticker
        def auto_detect_exchange():
            ticker = new_ticker.get().upper()
            if ticker.endswith('.SI'):
                new_exchange.set('SGX')
                new_country.set('Singapore')
            elif ticker.endswith('.KL'):
                new_exchange.set('KLSE')
                new_country.set('Malaysia')
            elif ticker.endswith('.HK'):
                new_exchange.set('HKEX')
                new_country.set('Hong Kong')
            elif ticker.endswith('.AX'):
                new_exchange.set('ASX')
                new_country.set('Australia')
            elif ticker.endswith('.TO'):
                new_exchange.set('TSX')
                new_country.set('Canada')
        
        # Bind auto-detection to ticker changes
        ticker_entry.bind('<KeyRelease>', lambda e: auto_detect_exchange())
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        apply_button = ttk.Button(button_frame, text="Apply Changes", 
                                 command=lambda: self.apply_company_changes(
                                     dialog, new_company_name.get(), new_ticker.get(), 
                                     new_industry.get(), new_country.get(), new_exchange.get()))
        apply_button.pack(side=tk.RIGHT, padx=(5, 0))
        
        cancel_button = ttk.Button(button_frame, text="Cancel", 
                                  command=dialog.destroy)
        cancel_button.pack(side=tk.RIGHT)
    
    def apply_preset(self, dialog, name, ticker, industry, country, exchange):
        """Apply a company preset"""
        try:
            # Update the config manager
            self.config_manager.update_company_info({
                'name': name,
                'ticker': ticker,
                'industry': industry,
                'country': country,
                'exchange': exchange
            })
            
            # Update the GUI variables
            self.company_name.set(name)
            self.ticker_symbol.set(ticker)
            self.country.set(country)
            self.stock_exchange.set(exchange)
            
            # Update the dialog variables
            dialog.destroy()
            
            self.log_message(f"Company preset applied: {name} ({ticker})")
            messagebox.showinfo("Success", f"Company configuration updated to {name} ({ticker})")
            
        except Exception as e:
            self.log_message(f"Error applying preset: {e}")
            messagebox.showerror("Error", f"Failed to apply preset: {e}")
    
    def apply_company_changes(self, dialog, company_name, ticker, industry, country, exchange):
        """Apply manual company changes"""
        try:
            # Validate inputs
            if not company_name or not ticker:
                messagebox.showerror("Error", "Company name and ticker symbol are required")
                return
            
            # Update the config manager
            self.config_manager.update_company_info({
                'name': company_name,
                'ticker': ticker,
                'industry': industry,
                'country': country,
                'exchange': exchange
            })
            
            # Update the GUI variables
            self.company_name.set(company_name)
            self.ticker_symbol.set(ticker)
            self.country.set(country)
            self.stock_exchange.set(exchange)
            
            # Close dialog
            dialog.destroy()
            
            self.log_message(f"Company configuration updated: {company_name} ({ticker})")
            messagebox.showinfo("Success", f"Company configuration updated to {company_name} ({ticker})")
            
        except Exception as e:
            self.log_message(f"Error applying company changes: {e}")
            messagebox.showerror("Error", f"Failed to update company configuration: {e}")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = ValueAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 