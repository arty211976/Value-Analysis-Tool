#!/usr/bin/env python3
"""
Setup script for Value Analysis Tool
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements-minimal.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="value-analysis-tool",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Powered Financial Analysis Tool using CrewAI",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/Value_Analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
            "mypy>=0.950",
        ],
        "full": [
            "torch>=1.9.0",
            "transformers>=4.0.0",
            "sentence-transformers>=2.0.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "value-analysis=Value_AnalysisGUI:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords="financial-analysis, ai, crewai, llm, investment, valuation, eps, roe",
    project_urls={
        "Bug Reports": "https://github.com/your-username/Value_Analysis/issues",
        "Source": "https://github.com/your-username/Value_Analysis",
        "Documentation": "https://github.com/your-username/Value_Analysis#readme",
    },
)