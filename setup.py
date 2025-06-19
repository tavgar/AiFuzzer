#!/usr/bin/env python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aifuzzer",
    version="0.1.0",
    author="AI Safety Researcher",
    author_email="researcher@example.com",
    description="A tool for testing LLM safety measures using Gemini to generate jailbreak prompts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aifuzzer/aifuzzer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "rich>=10.0.0",
        "google-generativeai>=0.3.0",
        "anthropic>=0.5.0",
        "pydantic>=1.8.0",
    ],
    entry_points={
        "console_scripts": [
            "aifuzzer=aifuzzer.main:main",
        ],
    },
)
